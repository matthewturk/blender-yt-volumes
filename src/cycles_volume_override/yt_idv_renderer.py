"""yt_idv volume renderers for the two-pass pipeline.

All modes share `_render_with_yt_idv` and differ only in how the
`block_rendering` component is configured.

Registration: the flagged object's `bounding_box_local`/`matrix_world` maps
yt_idv's unitary `[0, 1]^3` domain into Blender world space (unitary `u` ->
object-local `bbox_min + u*bbox_extent` -> `matrix_world`). The same mapping
positions the yt_idv camera and converts Cycles' Z-buffer into yt_idv window
depth for the external depth clip. A subset scene rescaled by
`scale_scene_to_subset` fills that cube along its longest axis, starting at
the origin corner; shorter axes leave empty space at their high end.

Zoom: `custom_params["yt_idv_zoom_target"]` (`"max"` or `"x, y, z"` unitary)
and `"yt_idv_zoom_level"` (log10 magnification) expand the data about the
target inside that cube, in float64. yt_idv's model space is then the cube's
coordinates around the target, rescaled each frame so the camera is about one
unit away; that keeps the GPU's float32 precise however deep the zoom, whether
the zoom comes from the level or from the Blender camera flying in. A camera
flying in needs the target near the world origin, where float32 is precise.

Spherical data: the volume object's inscribed sphere is the data's outer
radius, its local +z the pole (theta = 0) and +x longitude 0 (phi = 0).

Volume source: the `IsolatedGalaxy` sample by default.
`custom_params["yt_idv_volume_provider"]` (`"module.submodule:function"`)
names a callable `provider(ctx: VolumeSetupContext) -> VolumeSetupResult`,
called once per flagged object per frame. `ctx.previous_scene`/`previous_ds`
are what the provider last returned (None on the first call or when the
render context changed); return them with `changed=False` to reuse them.
`changed=True` forces a re-render. With `changed=False`, the previous
`RenderResult` is returned without calling `rc.run()` unless the camera,
depth buffer or `custom_params` changed. When unsure, return `changed=True`.

Render contexts, datasets, scenes and results are cached at module level and
never torn down.

Shading knobs are read from `custom_params` (`yt_idv_*` keys, filled from the
add-on Preferences by `operators.build_volume_params`), falling back to the
module constants below, which are tuned for IsolatedGalaxy density.

yt_idv gotchas handled here or in the required yt_idv patches:
  - Non-square renders show period-4 banding, so we render square and
    center-crop; the square's FOV covers the frame's longer side.
  - `SceneGraph.image` returns `(width, height, 4)` for non-square renders.
  - OSMesa corrupts non-square renders; the default backend is EGL.
  - The EGL config needs `EGL_ALPHA_SIZE` or backgrounds come back opaque.
  - `transfer_function` needs the `passthrough_alpha` second pass to keep its
    accumulated alpha.
  - Blender's GPU compositor holds an EGL context that makes yt_idv's
    `eglMakeCurrent` fail (`EGL_BAD_ACCESS`); `operators` forces the CPU
    compositor during yt_idv renders.
  - The external depth clip requires yt-project/yt_idv#247.
  - `scale_scene_to_subset` requires yt-project/yt_idv#287.

`yt`/`yt_idv` are imported lazily so the add-on registers without them.
"""

import dataclasses
import hashlib
import importlib
import inspect
import math

import numpy as np

from . import change_tracking
from .external_render import RenderRequest, RenderResult
from .sample_renderers import _persp_ortho_rays

SAMPLE_DATASET_NAME = "IsolatedGalaxy"
SAMPLE_FIELD = "density"
DEFAULT_BACKEND = "egl"  # OSMesa corrupts non-square renders

# --- Isocontour transfer function -----------------------------------------
# Values are in yt_idv's [0, 1] min-max-normalized space, not physical units.
# With `tf_log=True` the shader uses natural log, so `ColorTransferFunction`
# bounds use `math.log`, not `log10`.
TF_VALUE_MIN = 1e-6
TF_VALUE_MAX = 2e-3
TF_N_LAYERS = 5  # isocontour layers, evenly log-spaced between TF_VALUE_MIN/MAX
TF_COLORMAP = "viridis"
TF_NBINS = 256  # matches TransferFunctionTexture's expected (256, 1, 4) shape

# --- Threshold fog transfer function -----------------------------------------
# `transfer_function.frag.glsl` discards samples below `tf_min`, so
# TF_FOG_VALUE_MIN is the threshold; alpha ramps up above it.
TF_FOG_VALUE_MIN = 1e-9
TF_FOG_VALUE_MAX = 1e-2  # alpha reaches TF_FOG_ALPHA_MAX at this value
TF_FOG_COLORMAP = "dusk"
TF_FOG_ALPHA_MAX = 0.85  # < 1.0 keeps the fog soft at its densest
TF_FOG_RAMP_POWER = 0.15  # < 1 = alpha rises fast just past the threshold
# The transfer_function modes integrate opacity along the ray, so their value
# ranges are sensitive to camera geometry and need per-scene tuning.

# --- Max intensity with value-driven alpha ------------------------------------
# The ramp is a fraction of the per-view auto cmap_min/cmap_max range, so it
# needs no dataset-specific tuning.
MIP_ALPHA_COLORMAP = "dusk"
MIP_ALPHA_MAX = 0.7  # < 1.0 so objects behind the densest material still show through
MIP_ALPHA_RAMP_POWER = 0.15  # < 1 keeps more of the faint halo visible
MIP_ALPHA_NBINS = 256

SUBSET_BOUNDS_TOLERANCE = 1e-4  # model-space slack when checking scaled bounds
# With a zoom or shot frame the camera is ~1 model unit from the target; these
# bound the planes relative to that distance.
FAR_REACH = 1e3
NEAR_FRACTION = 1e-3

_dataset_cache = {}  # dataset name -> (ds, data_source)
_render_context_cache = {}  # (backend, width, height) -> rc
_volume_scene_cache = {}  # dataset_id -> (scene, ds, render_context_key)
_render_result_cache = {}  # dataset_id -> last RenderResult
_transfer_function_cache = {}  # (n_layers, colormap, value_min, value_max) -> (256, 1, 4) uint8 RGBA texture data
_fog_transfer_function_cache = {}  # (colormap, value_min, value_max, alpha_max, ramp_power) -> (256, 1, 4) uint8 RGBA texture data
_yt_tf_applied = {}  # id(component) -> (key, converted) of the yt transfer function it holds
_mip_alpha_colormap_cache = {}  # (colormap, alpha_max, ramp_power) -> (256, 4) float32 RGBA colormap data


def _load_sample_source():
    """Load the hardcoded sample dataset once per session."""
    cached = _dataset_cache.get(SAMPLE_DATASET_NAME)
    if cached is not None:
        return cached
    try:
        import yt
    except ImportError as exc:
        raise ImportError(
            "yt_idv rendering needs `yt` importable from Blender's own "
            "Python -- see README 'yt_idv integration' for how to install "
            "it there."
        ) from exc
    ds = yt.load_sample(SAMPLE_DATASET_NAME)
    data_source = ds.all_data()
    _dataset_cache[SAMPLE_DATASET_NAME] = (ds, data_source)
    return ds, data_source


def _get_render_context(backend, width, height):
    """Create (and cache) one offscreen render context per (backend, width,
    height); the framebuffer size is fixed at creation.
    """
    key = (backend, width, height)
    cached = _render_context_cache.get(key)
    if cached is not None:
        return cached
    try:
        import yt_idv
    except ImportError as exc:
        raise ImportError(
            "yt_idv rendering needs `yt_idv` importable from Blender's own "
            "Python -- see README 'yt_idv integration' for how to install "
            "it there."
        ) from exc
    rc = yt_idv.render_context(backend, width=width, height=height)
    _render_context_cache[key] = rc
    return rc


@dataclasses.dataclass
class VolumeSetupContext:
    """Passed to a `yt_idv_volume_provider` callable."""

    render_context: "object"  # live yt_idv render context; call .add_scene(...) on it
    dataset: "object"  # external_render.DatasetParams
    custom_params: dict  # same dict configure_component() receives
    frame: "object"  # external_render.FrameParams
    previous_scene: "object | None"  # your last returned `scene`, or None (first call, or render context changed)
    previous_ds: "object | None"  # your last returned `ds`, or None


@dataclasses.dataclass
class VolumeSetupResult:
    """Returned by a `yt_idv_volume_provider` callable."""

    scene: "object"  # what _render_with_yt_idv renders
    ds: "object"  # needs only `.domain_center`
    changed: bool  # True if the volume data changed; forces a re-render


def _resolve_volume_provider(spec):
    """Resolve a `"module.submodule:function_name"` string to a callable.

    Raises on a malformed spec rather than falling back to the default provider.
    """
    if not isinstance(spec, str) or ":" not in spec:
        raise ValueError(
            "custom_params['yt_idv_volume_provider'] must be a "
            f"'module.submodule:function_name' string, got {spec!r}"
        )
    module_path, _, func_name = spec.rpartition(":")
    module = importlib.import_module(module_path)
    try:
        return getattr(module, func_name)
    except AttributeError as exc:
        raise AttributeError(
            f"module {module_path!r} has no attribute {func_name!r} "
            f"(from custom_params['yt_idv_volume_provider']={spec!r})"
        ) from exc


def _default_volume_provider(ctx: VolumeSetupContext) -> VolumeSetupResult:
    """Default provider: the IsolatedGalaxy sample, built once."""
    if ctx.previous_scene is not None:
        return VolumeSetupResult(scene=ctx.previous_scene, ds=ctx.previous_ds, changed=False)
    ds, data_source = _load_sample_source()
    scene = ctx.render_context.add_scene(data_source, SAMPLE_FIELD, no_ghost=True)
    return VolumeSetupResult(scene=scene, ds=ds, changed=True)


def _block_collections(scene):
    """The `BlockCollection`s behind the scene's `block_rendering` components."""
    return [c.data for c in scene.components if getattr(c, "name", None) == "block_rendering"]


def _vertex_attribute(collection, name):
    for attribute in collection.vertex_array.attributes:
        if attribute.name == name:
            return np.asarray(attribute.data, dtype=np.float64)[:, :3]
    raise KeyError(name)


def _is_spherical(collection):
    return getattr(collection, "_yt_geom_str", "cartesian") == "spherical"


def _collection_bounds(collection):
    """The collection's visible extent in model space: its data source's bbox
    clipped to its blocks, or just the blocks if the bbox is unknown.
    """
    if _is_spherical(collection):
        # in_left_edge holds (r, theta, phi); the cartesian boxes are separate
        return (
            _vertex_attribute(collection, "le_cart").min(axis=0),
            _vertex_attribute(collection, "re_cart").max(axis=0),
        )
    low = _vertex_attribute(collection, "in_left_edge").min(axis=0)
    high = _vertex_attribute(collection, "in_right_edge").max(axis=0)
    bbox = getattr(collection, "bbox", None)
    if bbox is None:
        return low, high
    offset = np.asarray(collection.applied_scale_offset, dtype=np.float64)
    ratio = collection.applied_scale_ratio
    bbox_low, bbox_high = ((np.asarray(edge, dtype=np.float64) - offset) / ratio for edge in bbox)
    return np.maximum(low, bbox_low), np.minimum(high, bbox_high)


def _scene_model_bounds(scene):
    """`(low, high)` (3,) arrays bounding the scene's visible data in yt_idv
    model space, or None if the scene has no blocks.
    """
    bounds = [_collection_bounds(c) for c in _block_collections(scene)]
    if not bounds:
        return None
    return np.min([b[0] for b in bounds], axis=0), np.max([b[1] for b in bounds], axis=0)


def _fills_unit_cube(low, high, tol=SUBSET_BOUNDS_TOLERANCE):
    """True if `low` is the origin and the longest axis spans [0, 1]."""
    return bool(np.all(np.abs(low) <= tol) and abs(float((high - low).max()) - 1.0) <= tol)


def scale_scene_to_subset(scene):
    """Rescale a subset scene so its data source's longest axis spans yt_idv's
    [0, 1] model space, starting at the origin. Scenes that already fill it
    are left alone.
    """
    for collection in _block_collections(scene):
        if _is_spherical(collection):
            continue
        if not hasattr(collection, "apply_scale"):
            raise RuntimeError(
                "scale_scene_to_subset needs a yt_idv with scene transforms "
                "(yt-project/yt_idv#287); this install doesn't have it."
            )
        if collection.scale or _fills_unit_cube(*_collection_bounds(collection)):
            continue
        collection.scale = True


def _set_model_transform(collection, offset, ratio):
    """Make yt_idv model coordinates `(unitary - offset) / ratio`, replacing
    whatever transform the collection has now.
    """
    # BlockCollection only composes transforms, and `scale = False` is a no-op
    # when scale is already False, so compose the difference.
    current_offset = np.asarray(collection.applied_scale_offset, dtype=np.float64)
    current_ratio = collection.applied_scale_ratio
    if np.array_equal(current_offset, offset) and current_ratio == ratio:
        return
    collection.apply_scale((offset - current_offset) / current_ratio, ratio / current_ratio)


def _base_transform(collection):
    """The `(offset, ratio)` that maps the collection into the volume cube,
    recorded the first time a zoom changes it.
    """
    base = getattr(collection, "_cycles_yt_base_transform", None)
    if base is None:
        base = (np.asarray(collection.applied_scale_offset, dtype=np.float64), float(collection.applied_scale_ratio))
        collection._cycles_yt_base_transform = base
    return base


def _densest_cell_unitary(collection):
    """Center of the densest cell the bitmaps keep, from the block textures, as
    a float64 (3,) unitary position; for scenes loaded without their dataset.
    """
    # textures are vertex-centered: a cell's value is the mean of its 8 corners
    left_edges = _vertex_attribute(collection, "in_left_edge")
    widths = _vertex_attribute(collection, "in_dx")
    best_value, best_center = -np.inf, None
    for vbo_i, texture in collection.texture_objects.items():
        v = np.asarray(texture.data)
        if min(v.shape) < 2:
            continue
        cells = sum(
            v[i : v.shape[0] - 1 + i, j : v.shape[1] - 1 + j, k : v.shape[2] - 1 + k]
            for i in (0, 1) for j in (0, 1) for k in (0, 1)
        ) / 8.0
        cells = np.where(np.asarray(collection.bitmap_objects[vbo_i].data) > 0, cells, -np.inf)
        index = np.unravel_index(np.argmax(cells), cells.shape)
        if cells[index] > best_value:
            best_value = cells[index]
            best_center = left_edges[vbo_i] + (np.array(index) + 0.5) * widths[vbo_i]
    if best_center is None:
        raise ValueError("zoom target 'max': the scene has no cells")
    offset = np.asarray(collection.applied_scale_offset, dtype=np.float64)
    return best_center * collection.applied_scale_ratio + offset


def _zoom_target_unitary(spec, collection):
    """`"max"` or `"x, y, z"` as a float64 (3,) unitary position."""
    if spec.strip().lower() == "max":
        cached = getattr(collection, "_cycles_yt_zoom_max", None)
        if cached is None:
            if collection.data_source is None:
                cached = _densest_cell_unitary(collection)
            else:
                position = collection.data_source.argmax(collection.field)
                cached = np.array([float(p.to("unitary")) for p in position])
            collection._cycles_yt_zoom_max = cached
        return cached
    values = spec.replace(",", " ").split()
    if len(values) != 3:
        raise ValueError(f"zoom target {spec!r} is not 'max' or 'x, y, z'")
    return np.array([float(v) for v in values])


# yt_idv draws spherical data in cartesian coordinates scaled by the outer
# radius: a unit sphere at the model origin, the volume object's inscribed sphere.
SPHERICAL_MODEL_FRAME = (np.full(3, 0.5), 2.0)


def _shot_space_volume(scene, volume, custom_params):
    """`volume` registered in shot space when `custom_params` asks for it.

    Shot space (see `shot_space`) puts the zoom target at the world origin,
    `yt_idv_shot_space` world units per unitary length at zoom 0. The
    registration is built here in float64 and replaces the volume object's
    own float32 matrix, which plays no part.
    """
    world_scale = custom_params.get("yt_idv_shot_space")
    spec = custom_params.get("yt_idv_zoom_target")
    collections = _block_collections(scene)
    if not world_scale or not spec or not collections:
        return volume
    base_offset, base_ratio = _base_transform(collections[0])
    anchor = (_zoom_target_unitary(spec, collections[0]) - base_offset) / base_ratio
    local_min, local_extent = _bbox_local_min_extent(volume.bounding_box_local)
    per_cube = float(world_scale) * base_ratio  # world units per volume-cube width
    linear = per_cube / local_extent
    matrix = np.eye(4)
    matrix[:3, :3] = np.diag(linear)
    # world = per_cube * (cube - anchor), with cube = (local - local_min) / extent
    matrix[:3, 3] = -linear * local_min - per_cube * anchor
    return dataclasses.replace(volume, matrix_world=matrix.tolist())


def apply_zoom(scene, custom_params, volume=None, camera_world=None):
    """Set each block collection's transform for the zoom in `custom_params`.

    Returns the model frame `(anchor, scale)`: model = (cube - anchor) * scale,
    with `anchor` the zoom target in volume-cube coordinates. Given the volume
    and the camera's world position, `scale` puts the camera about one unit
    from the target, which only changes the numbers, not the image. Returns
    None for cartesian data without a zoom, and SPHERICAL_MODEL_FRAME for
    spherical data, which can't be zoomed.
    """
    spec = custom_params.get("yt_idv_zoom_target")
    if any(_is_spherical(c) for c in _block_collections(scene)):
        if spec:
            raise NotImplementedError("yt_idv can't rescale spherical data, so it can't be zoomed")
        return SPHERICAL_MODEL_FRAME
    if not spec:
        for collection in _block_collections(scene):
            base = getattr(collection, "_cycles_yt_base_transform", None)
            if base is not None:
                _set_model_transform(collection, *base)
        return None
    magnification = 10.0 ** float(custom_params.get("yt_idv_zoom_level", 0.0))
    collections = _block_collections(scene)
    if not collections:
        return None
    base_offset, base_ratio = _base_transform(collections[0])
    target = _zoom_target_unitary(spec, collections[0])
    anchor = (target - base_offset) / base_ratio
    scale = 1.0
    if volume is not None and camera_world is not None:
        A, b = _world_to_model(volume, (anchor, 1.0))
        distance = float(np.linalg.norm(A @ (np.asarray(camera_world, dtype=np.float64) - b)))
        scale = 1.0 / max(distance, 1e-300) if distance < 1.0 else 1.0
    for collection in collections:
        base_offset, base_ratio = _base_transform(collection)
        target = _zoom_target_unitary(spec, collection)
        _set_model_transform(collection, target, base_ratio / (magnification * scale))
    return anchor, scale


def replace_volume_data(scene, data_source, field, no_ghost=True):
    """Swap each block_rendering component's data for blocks built from
    `data_source`, keeping the scene, its camera and its shading settings, and
    free the old blocks' GPU memory.
    """
    from yt_idv.scene_data.block_collection import BlockCollection

    for component in scene.components:
        if getattr(component, "name", None) != "block_rendering":
            continue
        old = component.data
        new = BlockCollection(data_source=data_source)
        new.add_data(field, no_ghost=no_ghost)
        component.data = new
        scene.data_objects[scene.data_objects.index(old)] = new
        old.release()


def _get_volume_scene(request: RenderRequest, render_size: int):
    """Call this object's volume provider and cache its scene.

    Returns `(rc, scene, ds, changed)`.
    """
    custom_params = request.volume.custom_params or {}
    backend = custom_params.get("yt_idv_backend", DEFAULT_BACKEND)
    context_key = (backend, render_size, render_size)
    rc = _get_render_context(*context_key)

    provider_spec = custom_params.get("yt_idv_volume_provider")
    provider = _resolve_volume_provider(provider_spec) if provider_spec else _default_volume_provider

    dataset_id = request.volume.dataset.dataset_id
    cached = _volume_scene_cache.get(dataset_id)
    if cached is not None and cached[2] == context_key:
        previous_scene, previous_ds = cached[0], cached[1]
    else:
        # No cached scene, or it belongs to a different render context.
        previous_scene, previous_ds = None, None

    ctx = VolumeSetupContext(
        render_context=rc,
        dataset=request.volume.dataset,
        custom_params=custom_params,
        frame=request.frame,
        previous_scene=previous_scene,
        previous_ds=previous_ds,
    )
    result = provider(ctx)
    _volume_scene_cache[dataset_id] = (result.scene, result.ds, context_key)
    return rc, result.scene, result.ds, result.changed


def _render_signature(request: RenderRequest, render_size: int, render_mode: str = "") -> dict:
    """Everything besides the volume data that affects the image. The depth
    buffer is hashed by content, since occluders can move under a fixed camera.
    """
    return {
        "camera": dataclasses.asdict(request.camera),
        "depth_shape": list(request.depth.data.shape),
        "depth_hash": hashlib.sha256(np.ascontiguousarray(request.depth.data).tobytes()).hexdigest(),
        "render_size": render_size,
        "params_hash": request.volume.params_version.hash,
        "render_mode": render_mode,
    }


def _render_signature_changed(
    dataset_id: str, request: RenderRequest, render_size: int, volume_changed: bool, render_mode: str = ""
) -> bool:
    """True if the provider reported a change or the render signature differs
    from this object's last render.
    """
    signature = _render_signature(request, render_size, render_mode)
    version = change_tracking.version_for(dataset_id, "yt_idv_render_signature", signature)
    return volume_changed or version.changed


def _to_texture2d_axis_convention(depth):
    """Transform a top-down `(height, width)` array so `Texture2D` samples it at
    the expected screen position (`Texture2D` uploads `data.swapaxes(0, 1)` and
    `gl_FragCoord` is bottom-origin).
    """
    return depth[::-1, :].T


def set_external_depth_clip(scene, depth=None):
    """Enable the external depth clip on every component, or disable it with
    `depth=None`.

    `depth`: `(height, width)` float32, top-down rows, in yt_idv window-space
    depth (0..1, as `gl_FragDepth`), matching the square render size.

    Raises if `depth` is given but no component supports the clip (yt_idv
    without yt-project/yt_idv#247), since it would otherwise silently no-op.
    """
    try:
        from yt_idv.opengl_support import Texture2D
    except ImportError as exc:
        raise ImportError(
            "yt_idv rendering needs `yt_idv` importable from Blender's own "
            "Python -- see README 'yt_idv integration' for how to install "
            "it there."
        ) from exc
    any_component_supports_clip = False
    for component in scene.components:
        if not hasattr(component, "use_external_depth_clip"):
            continue  # e.g. annotation components that aren't block_rendering
        any_component_supports_clip = True
        if depth is None:
            component.use_external_depth_clip = False
            continue
        texture_data = _to_texture2d_axis_convention(np.asarray(depth, dtype="f4"))
        texture_data = np.ascontiguousarray(texture_data)
        # Reuse the Texture2D: yt_idv never frees GL textures, so a new one per
        # frame leaks GPU memory. Reassigning `.data` updates it in place.
        existing = component.external_depth_texture
        if isinstance(existing, Texture2D):
            existing.data = texture_data
        else:
            component.external_depth_texture = Texture2D(data=texture_data)
        component.use_external_depth_clip = True
    if depth is not None and not any_component_supports_clip:
        raise RuntimeError(
            "set_external_depth_clip(depth=...) was asked to enable the "
            "depth clip, but no component in this yt_idv scene has a "
            "`use_external_depth_clip` attribute -- this yt_idv install is "
            "missing the external-depth-clip changes (yt-project/"
            "yt_idv#247), so the clip would silently do nothing while every "
            "render still looked plausible. See this function's own "
            "docstring and README 'yt_idv integration' for how to get a "
            "yt_idv checkout that actually supports it."
        )


def _build_isocontour_transfer_function_data(value_min, value_max, n_layers, colormap, nbins=TF_NBINS):
    """`n_layers` isocontours evenly log-spaced between `value_min`/`value_max`
    (normalized values), via `ColorTransferFunction.add_layers`.

    Returns `(nbins, 1, 4)` uint8 RGBA for `TransferFunctionTexture.data`.
    """
    import yt

    tf = yt.ColorTransferFunction((math.log(value_min), math.log(value_max)), nbins=nbins)
    tf.add_layers(n_layers, colormap=colormap)
    rgba = np.stack([channel.y for channel in tf.funcs], axis=-1)  # (nbins, 4), [0, 1]
    return np.clip(rgba * 255.0, 0, 255).astype("u1").reshape(nbins, 1, 4)


def _get_isocontour_transfer_function_data(value_min, value_max, n_layers, colormap):
    """Cached `_build_isocontour_transfer_function_data`."""
    key = (n_layers, colormap, value_min, value_max)
    cached = _transfer_function_cache.get(key)
    if cached is not None:
        return cached
    data = _build_isocontour_transfer_function_data(value_min, value_max, n_layers, colormap)
    _transfer_function_cache[key] = data
    return data


def _threshold_fog_alpha_ramp(nbins, alpha_max, ramp_power):
    """`alpha_max * frac**ramp_power` for `frac` in [0, 1] over `nbins` bins.

    Bins are uniform in log value, since the shader applies `tf_log`.
    `ramp_power < 1` front-loads the ramp; `> 1` delays it.
    """
    frac = np.linspace(0.0, 1.0, nbins)
    return alpha_max * frac**ramp_power


def _resample_1d(values, nbins):
    """Linearly resample a 1D sequence onto `nbins` evenly spaced points."""
    values = np.asarray(values, dtype=np.float64)
    if values.shape[0] == nbins:
        return values
    src_x = np.linspace(0.0, 1.0, values.shape[0])
    dst_x = np.linspace(0.0, 1.0, nbins)
    return np.interp(dst_x, src_x, values)


def _resample_rgb(rgb, nbins):
    """Per-channel `_resample_1d` for an `(n, 3)` or `(n, 4)` array."""
    rgb = np.asarray(rgb, dtype=np.float64)
    if rgb.shape[0] == nbins:
        return rgb
    src_x = np.linspace(0.0, 1.0, rgb.shape[0])
    dst_x = np.linspace(0.0, 1.0, nbins)
    out = np.empty((nbins, rgb.shape[1]))
    for c in range(rgb.shape[1]):
        out[:, c] = np.interp(dst_x, src_x, rgb[:, c])
    return out


def _build_threshold_fog_transfer_function_data(
    value_min, value_max, colormap, alpha_max, ramp_power, nbins=TF_NBINS,
    alpha_curve=None, color_rgba=None,
):
    """Threshold-fog LUT: transparent below `value_min` (enforced by the shader's
    `tf_min` cutoff), alpha ramping up above it.

    Returns `(nbins, 1, 4)` uint8 RGBA for `TransferFunctionTexture.data`.

    `alpha_curve` (floats in [0, 1], scaled by `alpha_max`) and `color_rgba`
    (`(n, 3)` or `(n, 4)` in [0, 1]; any alpha column is ignored) optionally
    replace the ramp and the named colormap, independently.
    """
    if color_rgba is not None:
        rgb = _resample_rgb(color_rgba, nbins)[:, :3]
    else:
        import matplotlib
        import yt  # noqa: F401 -- import side effect: registers yt's own colormaps (e.g. "dusk") into matplotlib

        cmap = matplotlib.colormaps[colormap]
        frac = np.linspace(0.0, 1.0, nbins)
        rgb = cmap(frac)[:, :3]  # (nbins, 3); alpha comes from the ramp

    if alpha_curve is not None:
        alpha = _resample_1d(alpha_curve, nbins) * alpha_max
    else:
        alpha = _threshold_fog_alpha_ramp(nbins, alpha_max, ramp_power)

    rgba = np.concatenate([rgb, alpha[:, None]], axis=1)
    return np.clip(rgba * 255.0, 0, 255).astype("u1").reshape(nbins, 1, 4)


def _get_threshold_fog_transfer_function_data(
    value_min, value_max, colormap, alpha_max, ramp_power, alpha_curve=None, color_rgba=None,
):
    """Cached `_build_threshold_fog_transfer_function_data`; list arguments
    become tuples for the cache key.
    """
    alpha_key = tuple(alpha_curve) if alpha_curve is not None else None
    color_key = tuple(tuple(row) for row in color_rgba) if color_rgba is not None else None
    key = (colormap, value_min, value_max, alpha_max, ramp_power, alpha_key, color_key)
    cached = _fog_transfer_function_cache.get(key)
    if cached is not None:
        return cached
    data = _build_threshold_fog_transfer_function_data(
        value_min, value_max, colormap, alpha_max, ramp_power,
        alpha_curve=alpha_curve, color_rgba=color_rgba,
    )
    _fog_transfer_function_cache[key] = data
    return data


def _build_mip_alpha_colormap_data(
    colormap, alpha_max, ramp_power, nbins=MIP_ALPHA_NBINS, alpha_curve=None, color_rgba=None,
):
    """`(nbins, 4)` float32 RGBA for `ColormapTexture.data`, alpha ramped as in
    `_threshold_fog_alpha_ramp`.

    Position is relative to the per-view `cmap_min`/`cmap_max`, not a fixed
    value. `alpha_curve`/`color_rgba` as in
    `_build_threshold_fog_transfer_function_data`.
    """
    if color_rgba is not None:
        rgb = _resample_rgb(color_rgba, nbins)[:, :3]
    else:
        import matplotlib
        import yt  # noqa: F401 -- import side effect: registers yt's own colormaps (e.g. "dusk") into matplotlib

        cmap = matplotlib.colormaps[colormap]
        frac = np.linspace(0.0, 1.0, nbins)
        rgb = cmap(frac)[:, :3]

    if alpha_curve is not None:
        alpha = _resample_1d(alpha_curve, nbins) * alpha_max
    else:
        alpha = _threshold_fog_alpha_ramp(nbins, alpha_max, ramp_power)

    return np.concatenate([rgb, alpha[:, None]], axis=1).astype("f4")


def _get_mip_alpha_colormap_data(colormap, alpha_max, ramp_power, alpha_curve=None, color_rgba=None):
    """Cached `_build_mip_alpha_colormap_data`."""
    alpha_key = tuple(alpha_curve) if alpha_curve is not None else None
    color_key = tuple(tuple(row) for row in color_rgba) if color_rgba is not None else None
    key = (colormap, alpha_max, ramp_power, alpha_key, color_key)
    cached = _mip_alpha_colormap_cache.get(key)
    if cached is not None:
        return cached
    data = _build_mip_alpha_colormap_data(
        colormap, alpha_max, ramp_power, alpha_curve=alpha_curve, color_rgba=color_rgba,
    )
    _mip_alpha_colormap_cache[key] = data
    return data


def _approx_vertical_fov_deg(camera, frame_aspect, canvas_aspect=None):
    """Vertical FOV (degrees) of a canvas centered on a Blender frame that
    covers the whole frame; ignores pixel aspect.

    `frame_aspect` is the Blender frame's width / height, `canvas_aspect` the
    yt_idv canvas's (default the same).
    """
    if canvas_aspect is None:
        canvas_aspect = frame_aspect
    fit = camera.sensor_fit
    sensor = camera.sensor_width_mm
    if fit == "AUTO":
        # AUTO applies sensor_width to the frame's longer side
        fit = "HORIZONTAL" if frame_aspect >= 1.0 else "VERTICAL"
    elif fit == "VERTICAL":
        sensor = camera.sensor_height_mm
    half_tan = (sensor / 2.0) / camera.lens_mm
    if fit == "HORIZONTAL":
        tan_h, tan_v = half_tan, half_tan / frame_aspect
    else:
        tan_h, tan_v = half_tan * frame_aspect, half_tan
    return math.degrees(2.0 * math.atan(max(tan_v, tan_h / canvas_aspect)))


def _domain_array_unitary(value):
    """Convert `ds.domain_center`/`domain_width` (code_length) to a (3,) unitary
    float32 array. Plain arrays pass through unchanged.
    """
    if hasattr(value, "in_units"):
        value = value.in_units("unitary").d
    return np.asarray(value, dtype="f4")


def _bbox_local_min_extent(bounding_box_local):
    """(min, extent) of the object-local bounding box; zero-extent axes become
    1.0.
    """
    corners = np.asarray(bounding_box_local, dtype=np.float64)
    local_min = corners.min(axis=0)
    local_max = corners.max(axis=0)
    extent = local_max - local_min
    return local_min, np.where(extent == 0.0, 1.0, extent)


def _world_to_model(volume, model_frame=None):
    """`(A, b)` with yt_idv model coordinates `A @ (world - b)`.

    Without `model_frame` the model space is the volume cube's unitary
    coordinates. With `(anchor, scale)` it is `(cube - anchor) * scale`, and
    `b` is the anchor's world position, so points near it keep their float64
    precision however small their offsets are.
    """
    matrix = np.array(volume.matrix_world, dtype=np.float64)
    linear = matrix[:3, :3]
    local_min, local_extent = _bbox_local_min_extent(volume.bounding_box_local)
    anchor, scale = (np.zeros(3), 1.0) if model_frame is None else model_frame
    b = linear @ (local_min + np.asarray(anchor, dtype=np.float64) * local_extent) + matrix[:3, 3]
    A = (scale / local_extent)[:, None] * np.linalg.inv(linear)
    return A, b


def _registered_camera_kwargs(
    camera, volume, ds, width, height, model_bounds=None, model_frame=None, frame_aspect=None,
):
    """Map Blender `CameraParams` to `BaseCamera.update()` kwargs through the
    object bounding-box registration into yt_idv model space, for a `width` x
    `height` canvas covering a Blender frame of `frame_aspect` (default the
    canvas's).

    `model_bounds` (`(low, high)`, default the unit cube) sets the near/far
    planes. `model_frame` is `apply_zoom`'s; see `_world_to_model`.
    """
    if camera.projection != "PERSP":
        raise NotImplementedError(
            "yt_idv's camera is perspective-only (yt_idv.cameras."
            "trackball_camera.TrackballCamera has no ortho/panoramic "
            f"projection) -- got CameraParams.projection={camera.projection!r}."
        )

    cam_matrix_world = np.array(camera.matrix_world, dtype=np.float64)
    cam_pos_world = cam_matrix_world[:3, 3]
    # Blender camera-local forward is -Z, up is +Y.
    cam_forward_world = -cam_matrix_world[:3, 2]
    cam_up_world = cam_matrix_world[:3, 1]

    A, b = _world_to_model(volume, model_frame)
    position = A @ (cam_pos_world - b)

    # Directions use only the linear part of the transform.
    forward = A @ cam_forward_world
    forward_norm = np.linalg.norm(forward)
    if forward_norm == 0.0:
        raise ValueError(
            "camera forward direction vanished after the registration transform -- "
            "check VolumeObjectParams.matrix_world/bounding_box_local for degeneracy."
        )
    forward = forward / forward_norm

    up = A @ cam_up_world
    # Non-uniform extent can skew up vs. forward; re-orthogonalize.
    up = up - np.dot(up, forward) * forward
    up_norm = np.linalg.norm(up)
    up = (up / up_norm) if up_norm > 0.0 else np.array([0.0, 1.0, 0.0])

    if model_frame is None:
        domain_center = _domain_array_unitary(ds.domain_center).astype(np.float64)
        look_ahead = max(float(np.linalg.norm(position - domain_center)), 1e-3)
    else:
        # the distance to the zoom target, which is the model origin
        look_ahead = max(float(np.linalg.norm(position)), 1e-3)
    focus = position + forward * look_ahead

    aspect_ratio = width / height
    fov_deg = _approx_vertical_fov_deg(camera, frame_aspect or aspect_ratio, aspect_ratio)
    corners = _UNIT_CUBE_CORNERS if model_bounds is None else _box_corners(*model_bounds)
    planes = _near_far_planes(position, corners=corners)
    if model_frame is not None:
        # Deep in a zoom the domain is ~1e9 model units away; yt_idv rebuilds
        # rays from the inverse projection in float32, so bound far/near.
        to_target = float(np.linalg.norm(position))
        planes["far_plane"] = min(planes["far_plane"], FAR_REACH * to_target)
        planes["near_plane"] = min(max(planes["near_plane"], NEAR_FRACTION * to_target), planes["far_plane"] / 2)

    return {
        "position": position.astype("f4"),
        "focus": focus.astype("f4"),
        "up": up.astype("f4"),
        "fov": fov_deg,
        "aspect_ratio": aspect_ratio,
        **planes,
    }


_UNIT_CUBE_CORNERS = np.array(
    [[x, y, z] for x in (0.0, 1.0) for y in (0.0, 1.0) for z in (0.0, 1.0)],
    dtype=np.float64,
)


def _box_corners(low, high):
    low, high = np.asarray(low, dtype=np.float64), np.asarray(high, dtype=np.float64)
    return low + _UNIT_CUBE_CORNERS * (high - low)


def _near_far_planes(position, margin=0.1, corners=_UNIT_CUBE_CORNERS):
    """Near/far planes bracketing the box around `corners` (default the unit
    cube), plus `margin`; the near plane is floored inside the box.

    A loose near/far range squeezes the domain into a tiny window-depth band,
    making the external depth clip comparison flip randomly between pixels.
    """
    dists = np.linalg.norm(corners - position, axis=1)
    gap = np.maximum(np.maximum(corners.min(axis=0) - position, position - corners.max(axis=0)), 0.0)
    return {
        "near_plane": max(1e-5, float(np.linalg.norm(gap)) - margin),
        "far_plane": float(dists.max()) + margin,
    }


def _cycles_depth_to_yt_idv_window_depth(request, yt_idv_camera, model_frame=None):
    """Convert Cycles' PLANAR `request.depth` to yt_idv window-space depth.

    Reconstructs world hit points from the Blender camera rays, maps them into
    model space via the object registration, and projects them through
    `yt_idv_camera` (already positioned for this frame). No-hit pixels get 1.0.

    Returns a top-down `(height, width)` float32 array.
    """
    camera = request.camera
    depth_buffer = request.depth
    volume = request.volume

    origins, directions = _persp_ortho_rays(camera, depth_buffer.row_order, depth_buffer.data.shape)

    cam_matrix_world_inv = np.linalg.inv(np.array(camera.matrix_world, dtype=np.float64))
    cam_z_row = cam_matrix_world_inv[2, :3]
    cam_z_translation = cam_matrix_world_inv[2, 3]
    # PLANAR depth is -cam_z, and cam_z is affine in t, so solve for t directly.
    cam_z_origin = origins @ cam_z_row + cam_z_translation
    cam_z_direction = directions @ cam_z_row

    depth_data = depth_buffer.data.astype(np.float64)
    no_hit_mask = depth_data >= depth_buffer.no_hit_value * 0.999

    with np.errstate(divide="ignore", invalid="ignore"):
        t = (-depth_data - cam_z_origin) / cam_z_direction
    world_points = origins + t[..., None] * directions

    A, b = _world_to_model(volume, model_frame)
    h, w = world_points.shape[:2]
    model_points = (world_points - b) @ A.T

    proj_view = np.array(yt_idv_camera.projection_matrix, dtype=np.float64) @ np.array(
        yt_idv_camera.view_matrix, dtype=np.float64
    )
    model_points_h = np.concatenate([model_points, np.ones((h, w, 1))], axis=-1)
    with np.errstate(divide="ignore", invalid="ignore"):
        clip = model_points_h @ proj_view.T
        ndc_z = clip[..., 2] / clip[..., 3]
    window_depth = 0.5 * ndc_z + 0.5
    window_depth = np.where(no_hit_mask, 1.0, window_depth)
    window_depth = np.nan_to_num(window_depth, nan=1.0, posinf=1.0, neginf=1.0)
    window_depth = np.clip(window_depth, 0.0, 1.0).astype("f4")

    # Convert to top-down rows.
    if depth_buffer.row_order == "BOTTOM_LEFT_ORIGIN":
        window_depth = window_depth[::-1, :]
    elif depth_buffer.row_order != "TOP_LEFT_ORIGIN":
        raise ValueError(f"unknown row_order {depth_buffer.row_order!r}")
    return np.ascontiguousarray(window_depth)


def _pad_to_square(depth_2d, render_size, fill_value=1.0):
    """Center a `(height, width)` array in a square canvas filled with
    `fill_value`, using the same offsets as `_center_crop`.
    """
    height, width = depth_2d.shape
    canvas = np.full((render_size, render_size), fill_value, dtype="f4")
    y0 = (render_size - height) // 2
    x0 = (render_size - width) // 2
    canvas[y0 : y0 + height, x0 : x0 + width] = depth_2d
    return canvas


def _fix_yt_idv_axis_order(image, expected_height, expected_width):
    """Fix `SceneGraph.image` returning `(width, height, 4)` for non-square
    renders (its `swapaxes` result is discarded).
    """
    if image.shape[:2] == (expected_height, expected_width):
        return image
    if image.shape[:2] == (expected_width, expected_height):
        return image.swapaxes(0, 1)
    raise ValueError(
        f"yt_idv returned image shape {image.shape[:2]}, matching neither "
        f"requested (height, width)=({expected_height}, {expected_width}) "
        "nor its transpose -- something beyond the known axis-order bug "
        "(see module docstring) is going on."
    )


def _to_row_order(image, row_order):
    """Convert yt_idv's top-down image to `row_order`."""
    if row_order == "TOP_LEFT_ORIGIN":
        return image
    if row_order == "BOTTOM_LEFT_ORIGIN":
        return image[::-1, :, :]
    raise ValueError(f"unknown row_order {row_order!r}")


def _center_crop(image, expected_height, expected_width):
    """Center-crop a square render to `(expected_height, expected_width, 4)`."""
    size = image.shape[0]
    y0 = (size - expected_height) // 2
    x0 = (size - expected_width) // 2
    return image[y0 : y0 + expected_height, x0 : x0 + expected_width, :]


def _render_with_yt_idv(request: RenderRequest, configure_component) -> RenderResult:
    """Render `request` with yt_idv; `configure_component(component,
    custom_params)` sets up each scene component.

    Renders square and center-crops, padding the Cycles depth clip onto the
    same square canvas. Returns the cached result without `rc.run()` if
    nothing affecting the image changed.
    """
    height, width = request.depth.data.shape
    render_size = max(height, width)
    dataset_id = request.volume.dataset.dataset_id

    rc, scene, ds, volume_changed = _get_volume_scene(request, render_size)
    # The mode isn't in custom_params, so switching modes must be tracked here.
    render_mode = getattr(configure_component, "__qualname__", repr(configure_component))
    needs_render = _render_signature_changed(dataset_id, request, render_size, volume_changed, render_mode)
    if not needs_render:
        cached_result = _render_result_cache.get(dataset_id)
        if cached_result is not None:
            return cached_result
        # No cached result yet; render anyway.

    # rc.run() renders rc.scene, which another object sharing this context may
    # have replaced.
    rc.scene = scene

    custom_params = request.volume.custom_params or {}
    request = dataclasses.replace(request, volume=_shot_space_volume(scene, request.volume, custom_params))
    camera_world = np.array(request.camera.matrix_world, dtype=np.float64)[:3, 3]
    model_frame = apply_zoom(scene, custom_params, request.volume, camera_world)
    context = {
        "frame": request.frame.frame_number,
        "camera": _camera_unitary(scene, request.volume, model_frame, camera_world),
        "ds": ds,
    }
    for component in scene.components:
        configure_component(component, dict(custom_params, _context=context))
    scene.camera.update(
        **_registered_camera_kwargs(
            request.camera, request.volume, ds, render_size, render_size,
            model_bounds=_scene_model_bounds(scene), model_frame=model_frame,
            frame_aspect=width / height,
        )
    )

    window_depth = _cycles_depth_to_yt_idv_window_depth(request, scene.camera, model_frame)
    set_external_depth_clip(scene, _pad_to_square(window_depth, render_size))

    image = np.asarray(rc.run(), dtype=np.float32)  # nominally (render_size, render_size, 4), [0, 1]
    image = _fix_yt_idv_axis_order(image, render_size, render_size)  # defensive; square shape makes this a no-op
    image = _to_row_order(image, request.depth.row_order)
    image = _center_crop(image, height, width)

    result = RenderResult(
        rgba=np.ascontiguousarray(image),
        alpha_mode="PREMULTIPLIED",  # approximation
        color_space="LINEAR",
    )
    _render_result_cache[dataset_id] = result
    return result


def _configure_max_intensity(component, custom_params):
    component.render_method = "max_intensity"


def _apply_transfer_function_settings(component, tf_min, tf_max, tf_data):
    """Apply transfer-function settings; split out so it's testable without
    `yt`.
    """
    component.tf_min = tf_min
    component.tf_max = tf_max
    component.tf_log = True
    component.transfer_function.data = tf_data


def _configure_transfer_function(component, custom_params):
    """Isocontour transfer function; `yt_idv_tf_*` custom_params override the
    `TF_*` defaults.
    """
    component.render_method = "transfer_function"
    value_min = custom_params.get("yt_idv_tf_value_min", TF_VALUE_MIN)
    value_max = custom_params.get("yt_idv_tf_value_max", TF_VALUE_MAX)
    n_layers = int(custom_params.get("yt_idv_tf_n_layers", TF_N_LAYERS))
    colormap = custom_params.get("yt_idv_tf_colormap", TF_COLORMAP)
    _apply_transfer_function_settings(
        component, value_min, value_max,
        _get_isocontour_transfer_function_data(value_min, value_max, n_layers, colormap),
    )


def yt_idv_max_intensity_renderer(request: RenderRequest) -> RenderResult:
    """`RENDERER_MODES["YT_IDV_MAX_INTENSITY"]`."""
    return _render_with_yt_idv(request, _configure_max_intensity)


def _configure_max_intensity_alpha(component, custom_params):
    """Max intensity with value-driven alpha; `yt_idv_mip_alpha_*`,
    `yt_idv_alpha_curve` and `yt_idv_color_image_rgba` custom_params override
    the defaults.
    """
    component.render_method = "max_intensity_alpha"
    colormap = custom_params.get("yt_idv_mip_alpha_colormap", MIP_ALPHA_COLORMAP)
    alpha_max = custom_params.get("yt_idv_mip_alpha_max", MIP_ALPHA_MAX)
    ramp_power = custom_params.get("yt_idv_mip_alpha_ramp_power", MIP_ALPHA_RAMP_POWER)
    alpha_curve = custom_params.get("yt_idv_alpha_curve")
    color_rgba = custom_params.get("yt_idv_color_image_rgba")
    component.colormap.data = _get_mip_alpha_colormap_data(
        colormap, alpha_max, ramp_power, alpha_curve=alpha_curve, color_rgba=color_rgba,
    )


def yt_idv_max_intensity_alpha_renderer(request: RenderRequest) -> RenderResult:
    """`RENDERER_MODES["YT_IDV_MAX_INTENSITY_ALPHA"]`: MIP with alpha driven by
    the selected value instead of a binary hit flag.
    """
    return _render_with_yt_idv(request, _configure_max_intensity_alpha)


def yt_idv_transfer_function_renderer(request: RenderRequest) -> RenderResult:
    """`RENDERER_MODES["YT_IDV_TRANSFER_FUNCTION"]`: isocontour layers in log
    space.
    """
    return _render_with_yt_idv(request, _configure_transfer_function)


def _configure_threshold_fog(component, custom_params):
    """Threshold fog; `yt_idv_fog_*`, `yt_idv_alpha_curve` and
    `yt_idv_color_image_rgba` custom_params override the defaults.
    """
    component.render_method = "transfer_function"
    value_min = custom_params.get("yt_idv_fog_value_min", TF_FOG_VALUE_MIN)
    value_max = custom_params.get("yt_idv_fog_value_max", TF_FOG_VALUE_MAX)
    colormap = custom_params.get("yt_idv_fog_colormap", TF_FOG_COLORMAP)
    alpha_max = custom_params.get("yt_idv_fog_alpha_max", TF_FOG_ALPHA_MAX)
    ramp_power = custom_params.get("yt_idv_fog_ramp_power", TF_FOG_RAMP_POWER)
    alpha_curve = custom_params.get("yt_idv_alpha_curve")
    color_rgba = custom_params.get("yt_idv_color_image_rgba")
    _apply_transfer_function_settings(
        component, value_min, value_max,
        _get_threshold_fog_transfer_function_data(
            value_min, value_max, colormap, alpha_max, ramp_power,
            alpha_curve=alpha_curve, color_rgba=color_rgba,
        ),
    )


def yt_idv_threshold_fog_renderer(request: RenderRequest) -> RenderResult:
    """`RENDERER_MODES["YT_IDV_THRESHOLD_FOG"]`: transparent below
    `TF_FOG_VALUE_MIN`, alpha ramping up to `TF_FOG_ALPHA_MAX`.
    """
    return _render_with_yt_idv(request, _configure_threshold_fog)


def _resolve_callable(spec):
    """`"module:function"` or `"path/to/file.py:function"` as a callable."""
    target, sep, name = spec.rpartition(":")
    if not sep or not target:
        raise ValueError(f"{spec!r} is not 'module:function' or 'file.py:function'")
    if target.endswith(".py"):
        import importlib.util
        import os

        module_spec = importlib.util.spec_from_file_location(os.path.splitext(os.path.basename(target))[0], target)
        module = importlib.util.module_from_spec(module_spec)
        module_spec.loader.exec_module(module)
    else:
        module = importlib.import_module(target)
    return getattr(module, name)


def _camera_unitary(scene, volume, model_frame, camera_world):
    """The camera's position in the dataset's unitary coordinates, float64,
    or None for a scene without blocks.
    """
    collections = _block_collections(scene)
    if not collections or _is_spherical(collections[0]):
        return None
    A, b = _world_to_model(volume, model_frame)
    model = A @ (camera_world - b)
    collection = collections[0]
    return model * collection.applied_scale_ratio + np.asarray(collection.applied_scale_offset, dtype=np.float64)


def _configure_yt_transfer_function(component, custom_params):
    """A yt ColorTransferFunction from `yt_idv_yt_tf`, converted for this
    component's blocks with yt_idv.tf_conversion (yt-project/yt_idv#288).

    A function taking no arguments is converted once per set of blocks and
    settings. One taking an argument gets a context dict every frame: `frame`,
    `camera` (unitary float64 position) and `ds`.
    """
    from yt_idv.tf_conversion import apply_converted_transfer_function, convert_yt_transfer_function

    spec = custom_params.get("yt_idv_yt_tf")
    if not spec:
        raise ValueError("the yt Transfer Function mode needs yt_idv_yt_tf (Preferences > yt Transfer Function)")
    sample_factor = float(custom_params.get("yt_idv_yt_tf_sample_factor", 0.0))
    if sample_factor > 0:
        component.sample_factor = sample_factor
    ray_length = float(custom_params.get("yt_idv_yt_tf_ray_length", 1.0))
    make = _resolve_callable(spec)
    per_frame = bool(inspect.signature(make).parameters)
    key = (id(component.data), spec, ray_length)
    applied = _yt_tf_applied.get(id(component))
    if applied is not None and applied[0] == key and not per_frame:
        return
    tf = make(custom_params.get("_context", {})) if per_frame else make()
    # thin yt layers need a finer table than the 1024-texel default
    converted = convert_yt_transfer_function(tf, component.data, ray_length, nbins=4096)
    apply_converted_transfer_function(component, converted)
    if applied is not None:
        applied[1]["transfer_function"].release()
    _yt_tf_applied[id(component)] = (key, converted)


def yt_idv_yt_transfer_function_renderer(request: RenderRequest) -> RenderResult:
    """`RENDERER_MODES["YT_IDV_YT_TRANSFER_FUNCTION"]`: a yt-built transfer function."""
    return _render_with_yt_idv(request, _configure_yt_transfer_function)


def _configure_projection(component, custom_params):
    component.render_method = "projection"


def yt_idv_projection_renderer(request: RenderRequest) -> RenderResult:
    """`RENDERER_MODES["YT_IDV_PROJECTION"]`: additive line-of-sight
    integration.
    """
    return _render_with_yt_idv(request, _configure_projection)
