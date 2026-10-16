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

Volume source: the object's Dataset Path (a yt dataset, or a scene saved with
SceneGraph.save), else the `IsolatedGalaxy` sample.
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
object's Volume Override properties by `operators.build_volume_params`), falling back to the
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
import types

import numpy as np

from . import change_tracking
from .external_render import RenderRequest, RenderResult
from .sample_renderers import _persp_ortho_rays

SAMPLE_DATASET_NAME = "IsolatedGalaxy"
SAMPLE_FIELD = "density"
DEFAULT_BACKEND = "egl"  # OSMesa corrupts non-square renders
# The context's own surface; every render draws into framebuffers of its own size.
CONTEXT_SURFACE_SIZE = 64

# --- yt layers transfer function ---------------------------------------------
# Bounds are log10 of the field in its own units, as for a yt ColorTransferFunction.
TF_LOG_MIN = -28.0
TF_LOG_MAX = -22.0
TF_N_LAYERS = 8
TF_COLORMAP = "arbre"
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
_render_context_cache = {}  # backend -> rc
_volume_scene_cache = {}  # dataset_id -> (scene, ds, backend)
_render_result_cache = {}  # (dataset_id, render_size) -> last RenderResult
_depth_textures = {}  # (id(component), shape) -> its external depth-clip Texture2D
_fog_transfer_function_cache = {}  # (colormap, value_min, value_max, alpha_max, ramp_power) -> (256, 1, 4) uint8 RGBA texture data
_DISTRIBUTION_BINS = 2048  # log10 bins behind percentiles and the per-point distribution
_gn_tf_textures = {}  # id(component) -> its float32 TransferFunctionTexture for the GN mode
_yt_tf_applied = {}  # id(component) -> (key, converted) of the yt transfer function it holds
_mip_alpha_colormap_cache = {}  # (colormap, alpha_max, ramp_power) -> (256, 4) float32 RGBA colormap data


def load_sample(name, online_access=None, **kwargs):
    """`yt.load_sample`, but downloading only if Blender allows online access
    (`online_access`, default `bpy.app.online_access`).
    """
    import yt
    from yt.sample_data.api import _get_test_data_dir_path

    # load_sample only goes online when <test_data_dir>/<top directory> is missing.
    on_disk = _get_test_data_dir_path().joinpath(name.split("/")[0]).exists()
    if not on_disk and online_access is None:
        import bpy

        online_access = bpy.app.online_access
    if not on_disk and not online_access:
        raise RuntimeError(
            f"yt sample dataset {name!r} isn't downloaded and Blender's online "
            "access is off. Enable Preferences > System > Network > Allow Online "
            "Access, or on the command line pass --online-mode to blender or "
            "--online to render_volume."
        )
    return yt.load_sample(name, **kwargs)


def _load_sample_source(online_access=None):
    """Load the hardcoded sample dataset once per session."""
    cached = _dataset_cache.get(SAMPLE_DATASET_NAME)
    if cached is not None:
        return cached
    try:
        import yt  # noqa: F401
    except ImportError as exc:
        raise ImportError(
            "yt_idv rendering needs `yt` importable from Blender's own "
            "Python -- see docs/installing-in-blender.md."
        ) from exc
    ds = load_sample(SAMPLE_DATASET_NAME, online_access)
    data_source = ds.all_data()
    _dataset_cache[SAMPLE_DATASET_NAME] = (ds, data_source)
    return ds, data_source


def _get_render_context(backend):
    """Create (and cache) one offscreen render context per backend; renders
    of any size share it, and the scenes loaded into it (see `_use_render_size`).
    """
    key = backend
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
    rc = yt_idv.render_context(backend, width=CONTEXT_SURFACE_SIZE, height=CONTEXT_SURFACE_SIZE)
    _render_context_cache[key] = rc
    return rc


def _unorm16_texture(size):
    """A `size` x `size` RGBA texture stored as GL_RGBA16: clamped to [0, 1] on
    every write, as projection's summed color and alpha need, without 8-bit's
    rounding of faint linear values (bands, lost gas).
    """
    import traitlets
    from OpenGL import GL
    from yt_idv.opengl_support import Texture2D

    class Unorm16Texture(Texture2D):
        @traitlets.observe("data")
        def _set_data(self, change):
            data = change["new"]
            with self.bind():
                GL.glTexStorage2D(GL.GL_TEXTURE_2D, 1, GL.GL_RGBA16, data.shape[0], data.shape[1])
                for name, value in (
                    (GL.GL_TEXTURE_WRAP_S, self.boundary_x), (GL.GL_TEXTURE_WRAP_T, self.boundary_y),
                    (GL.GL_TEXTURE_MIN_FILTER, GL.GL_NEAREST), (GL.GL_TEXTURE_MAG_FILTER, GL.GL_NEAREST),
                ):
                    GL.glTexParameteri(GL.GL_TEXTURE_2D, name, value)

    return Unorm16Texture(data=np.zeros((size, size, 4), "u1"), boundary_x="repeat", boundary_y="repeat")


def _use_render_size(scene, size):
    """Make the next `rc.run()` render `scene` as a `size` x `size` image: set
    the viewport and give the scene and its components framebuffers that size.

    A framebuffer's size is fixed when first used, so a new size gets new ones;
    a component frees its old one when `fb` is reassigned, the scene doesn't.
    """
    from OpenGL import GL
    from yt_idv.opengl_support import Framebuffer

    def sized(fb_tex=None):
        fb = Framebuffer() if fb_tex is None else Framebuffer(fb_tex=fb_tex)
        fb._yt_volumes_size = size
        return fb

    GL.glViewport(0, 0, size, size)
    if getattr(scene.fb, "_yt_volumes_size", None) != size:
        old, scene.fb = scene.fb, sized(_unorm16_texture(size))
        if old is not None:
            old.release()
    for element in scene:
        if getattr(element.fb, "_yt_volumes_size", None) != size:
            element.fb = sized()


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


def _parse_field(spec):
    """`"density"` or `"gas, density"` as a yt field name or tuple."""
    parts = [p.strip() for p in str(spec).split(",")]
    return tuple(parts) if len(parts) == 2 else parts[0]


def _default_volume_provider(ctx: VolumeSetupContext) -> VolumeSetupResult:
    """Default provider: the object's Dataset Path, built once.

    A `.zip` (or kind `yt_idv_scene`) is a scene saved with SceneGraph.save,
    scaled so its saved bbox fills the volume box; anything else goes to
    yt.load, rendering custom_params `yt_idv_field` (default density). Without
    a path, the IsolatedGalaxy sample.
    """
    path = ctx.dataset.dataset_path
    source = path or SAMPLE_DATASET_NAME
    previous = ctx.previous_scene
    if previous is not None:
        if getattr(previous, "_yt_volumes_source", None) == source:
            return VolumeSetupResult(scene=previous, ds=ctx.previous_ds, changed=False)
        for collection in _block_collections(previous):
            collection.release()
    if not path:
        ds, data_source = _load_sample_source(ctx.custom_params.get("yt_idv_online_access"))
        scene = ctx.render_context.add_scene(data_source, SAMPLE_FIELD, no_ghost=True)
    elif path.endswith(".zip") or ctx.dataset.dataset_kind == "yt_idv_scene":
        from yt_idv.scene_graph import SceneGraph

        scene = SceneGraph.load(path)
        scale_scene_to_subset(scene)
        # the renderer reads only ds.domain_center, in unitary units
        ds = types.SimpleNamespace(domain_center=np.full(3, 0.5))
    else:
        import yt

        ds = yt.load(path)
        field = _parse_field(ctx.custom_params.get("yt_idv_field", SAMPLE_FIELD))
        scene = ctx.render_context.add_scene(ds.all_data(), field, no_ghost=True)
    scene._yt_volumes_source = source
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
    base = getattr(collection, "_yt_volumes_base_transform", None)
    if base is None:
        base = (np.asarray(collection.applied_scale_offset, dtype=np.float64), float(collection.applied_scale_ratio))
        collection._yt_volumes_base_transform = base
    return base


def _densest_cell_unitary(collection):
    """Center of the densest cell the bitmaps keep, from the block data, as
    a float64 (3,) unitary position; for scenes loaded without their dataset.
    """
    # textures are vertex-centered: a cell's value is the mean of its 8 corners
    left_edges = _vertex_attribute(collection, "in_left_edge")
    widths = _vertex_attribute(collection, "in_dx")
    best_value, best_center = -np.inf, None
    for vbo_i, data in collection.block_data.items():
        v = np.asarray(data)
        if min(v.shape) < 2:
            continue
        cells = sum(
            v[i : v.shape[0] - 1 + i, j : v.shape[1] - 1 + j, k : v.shape[2] - 1 + k]
            for i in (0, 1) for j in (0, 1) for k in (0, 1)
        ) / 8.0
        cells = np.where(np.asarray(collection.block_bitmaps[vbo_i]) > 0, cells, -np.inf)
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
        cached = getattr(collection, "_yt_volumes_zoom_max", None)
        if cached is None:
            if collection.data_source is None:
                cached = _densest_cell_unitary(collection)
            else:
                position = collection.data_source.argmax(collection.field)
                cached = np.array([float(p.to("unitary")) for p in position])
            collection._yt_volumes_zoom_max = cached
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
            base = getattr(collection, "_yt_volumes_base_transform", None)
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
    context_key = custom_params.get("yt_idv_backend", DEFAULT_BACKEND)
    rc = _get_render_context(context_key)

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


def needs_load(request: RenderRequest) -> bool:
    """True if rendering `request` will load its volume rather than reuse the
    cached scene: a first render, another dataset, or another backend.
    """
    cached = _volume_scene_cache.get(request.volume.dataset.dataset_id)
    if cached is None:
        return True
    backend = (request.volume.custom_params or {}).get("yt_idv_backend", DEFAULT_BACKEND)
    source = request.volume.dataset.dataset_path or SAMPLE_DATASET_NAME
    return cached[2] != backend or getattr(cached[0], "_yt_volumes_source", source) != source


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
    from this object's last successful render.
    """
    signature = _render_signature(request, render_size, render_mode)
    version = change_tracking.version_for(
        _sized_key(dataset_id, render_size), "yt_idv_render_signature", signature, record=False,
    )
    return volume_changed or version.changed


def _record_render_signature(dataset_id: str, request: RenderRequest, render_size: int, render_mode: str = ""):
    signature = _render_signature(request, render_size, render_mode)
    change_tracking.version_for(_sized_key(dataset_id, render_size), "yt_idv_render_signature", signature)


def _sized_key(dataset_id, render_size):
    # F12 and viewport renders of one dataset are tracked apart, so neither invalidates the other.
    return f"{dataset_id}@{render_size}"


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
        # Reuse one Texture2D per size: yt_idv never frees GL textures, so a new
        # one per frame leaks GPU memory, and a texture's size is fixed by its
        # first upload. Reassigning `.data` updates it in place.
        key = (id(component), texture_data.shape)
        existing = _depth_textures.get(key)
        if existing is not None:
            existing.data = texture_data
        else:
            existing = _depth_textures[key] = Texture2D(data=texture_data)
        component.external_depth_texture = existing
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
    planes = camera_planes(position, model_bounds, model_frame)

    return {
        "position": position.astype("f4"),
        "focus": focus.astype("f4"),
        "up": up.astype("f4"),
        "fov": fov_deg,
        "aspect_ratio": aspect_ratio,
        **planes,
    }


def camera_planes(position, model_bounds=None, model_frame=None):
    """Near/far planes for a camera at model `position` around `model_bounds`
    (default the unit cube); see `_registered_camera_kwargs`.
    """
    corners = _UNIT_CUBE_CORNERS if model_bounds is None else _box_corners(*model_bounds)
    planes = _near_far_planes(np.asarray(position, dtype=np.float64), corners=corners)
    if model_frame is not None:
        # Deep in a zoom the domain is ~1e9 model units away; yt_idv rebuilds
        # rays from the inverse projection in float32, so bound far/near.
        to_target = float(np.linalg.norm(position))
        planes["far_plane"] = min(planes["far_plane"], FAR_REACH * to_target)
        planes["near_plane"] = min(max(planes["near_plane"], NEAR_FRACTION * to_target), planes["far_plane"] / 2)
    return planes


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
    """Convert the PLANAR `request.depth` to yt_idv window-space depth.

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

    Renders square and center-crops, padding the depth clip onto the
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
        cached_result = _render_result_cache.get((dataset_id, render_size))
        if cached_result is not None:
            return cached_result
        # No cached result yet; render anyway.
    # Dropped up front so a render that raises can't leave a stale result to reuse.
    _render_result_cache.pop((dataset_id, render_size), None)

    # rc.run() renders rc.scene, which another object sharing this context may
    # have replaced.
    rc.scene = scene

    custom_params = request.volume.custom_params or {}
    signed_request = request
    request = dataclasses.replace(request, volume=_shot_space_volume(scene, request.volume, custom_params))
    camera_world = np.array(request.camera.matrix_world, dtype=np.float64)[:3, 3]
    model_frame = apply_zoom(scene, custom_params, request.volume, camera_world)
    camera_unitary = _camera_unitary(scene, request.volume, model_frame, camera_world)
    context = {
        "frame": request.frame.frame_number,
        "camera": camera_unitary,
        "value": None if camera_unitary is None else _field_value_at(scene, camera_unitary),
        "ds": ds,
    }
    for component in scene.components:
        _apply_samples_per_cell(component, custom_params)
        configure_component(component, dict(custom_params, _context=context))
        _apply_colormap_range(component, custom_params)
    scene.camera.update(
        **_registered_camera_kwargs(
            request.camera, request.volume, ds, render_size, render_size,
            model_bounds=_scene_model_bounds(scene), model_frame=model_frame,
            frame_aspect=width / height,
        )
    )

    window_depth = _cycles_depth_to_yt_idv_window_depth(request, scene.camera, model_frame)
    set_external_depth_clip(scene, _pad_to_square(window_depth, render_size))

    _use_render_size(scene, render_size)
    image = np.asarray(rc.run(), dtype=np.float32)  # nominally (render_size, render_size, 4), [0, 1]
    image = _fix_yt_idv_axis_order(image, render_size, render_size)  # defensive; square shape makes this a no-op
    image = _to_row_order(image, request.depth.row_order)
    image = _center_crop(image, height, width)

    result = RenderResult(
        rgba=np.ascontiguousarray(image),
        alpha_mode="PREMULTIPLIED",  # approximation
        color_space="LINEAR",
        cmap_range=_component_cmap_range(scene),
    )
    _render_result_cache[(dataset_id, render_size)] = result
    _record_render_signature(dataset_id, signed_request, render_size, render_mode)
    return result


def _apply_samples_per_cell(component, custom_params):
    """Set the component's ray sampling from `yt_idv_samples_per_cell`."""
    samples = custom_params.get("yt_idv_samples_per_cell")
    if samples is None or not hasattr(component, "sample_factor"):
        return
    samples = float(samples)
    if _is_spherical(component.data):
        # spherical sample_factor is log10 of the step over the element's smallest width
        component.sample_factor = -math.log10(samples)
    else:
        component.sample_factor = samples


def _apply_colormap_range(component, custom_params):
    """Refit the colormap range this render, or pin it to the volume's
    remembered `yt_idv_cmap_range`. Fixed bounds (`fixed_cmap_min/max`) win.
    """
    remembered = custom_params.get("yt_idv_cmap_range")
    if custom_params.get("yt_idv_rescale_colormap") or remembered is None:
        component._cmap_bounds_invalid = True
    elif component.fixed_cmap_min is None and component.fixed_cmap_max is None:
        component.cmap_min, component.cmap_max = (float(v) for v in remembered)
        component._cmap_bounds_invalid = False


def _component_cmap_range(scene):
    """`(min, max)` of the first component with a colormap range, or None."""
    for component in scene.components:
        low, high = getattr(component, "cmap_min", None), getattr(component, "cmap_max", None)
        if low is not None and high is not None:
            return (float(low), float(high))
    return None


def _set_colormap_name(component, custom_params):
    name = custom_params.get("yt_idv_colormap")
    if name and component.colormap.colormap_name != name:
        component.colormap.colormap_name = name


def _configure_max_intensity(component, custom_params):
    component.render_method = "max_intensity"
    _set_colormap_name(component, custom_params)


def _apply_transfer_function_settings(component, tf_min, tf_max, tf_data):
    """Apply transfer-function settings; split out so it's testable without
    `yt`.
    """
    component.tf_min = tf_min
    component.tf_max = tf_max
    component.tf_log = True
    component.transfer_function.data = tf_data


def _layer_bounds(collection, custom_params):
    """`(low, high)` log10 bounds of the layers transfer function: the data
    between `yt_idv_tf_percentile_low`/`_high` (by volume), or without
    `yt_idv_tf_fit_range` `yt_idv_tf_log_min`/`_max`.
    """
    distribution = _value_distribution(collection)
    if distribution is not None and custom_params.get("yt_idv_tf_fit_range", True):
        low, high = _percentile_range(
            distribution,
            custom_params.get("yt_idv_tf_percentile_low", 50.0),
            custom_params.get("yt_idv_tf_percentile_high", 100.0),
        )
        return tuple(float(np.log10(_physical(collection, v))) for v in (low, high))
    return (float(custom_params.get("yt_idv_tf_log_min", TF_LOG_MIN)),
            float(custom_params.get("yt_idv_tf_log_max", TF_LOG_MAX)))


def _configure_transfer_function(component, custom_params):
    """yt's `ColorTransferFunction.add_layers` from the `yt_idv_tf_*`
    custom_params, converted as for the yt Transfer Function mode.
    """
    get = custom_params.get
    bounds = _layer_bounds(component.data, custom_params)
    component.data._yt_volumes_layer_bounds = bounds
    n_layers = int(get("yt_idv_tf_n_layers", TF_N_LAYERS))
    width = float(get("yt_idv_tf_layer_width", 0.0)) or None  # 0: yt's default
    alpha = tuple(np.geomspace(float(get("yt_idv_tf_alpha_first", 1.0)), float(get("yt_idv_tf_alpha_last", 1.0)), n_layers))
    colormap = get("yt_idv_tf_colormap", TF_COLORMAP)

    def make():
        import yt

        tf = yt.ColorTransferFunction(bounds, nbins=8192, grey_opacity=True)
        tf.add_layers(n_layers, w=width, alpha=alpha, colormap=colormap)
        return tf

    _apply_yt_transfer_function(
        component, custom_params, ("layers", bounds, n_layers, width, alpha, colormap), make,
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
    """`RENDERER_MODES["YT_IDV_TRANSFER_FUNCTION"]`: yt's `add_layers`,
    evenly spaced in log10 of the field.
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


def _positive_value_range(collection):
    """The smallest positive and the largest texel value in `collection`'s
    blocks, cached on it; None if none is positive.
    """
    cached = getattr(collection, "_yt_volumes_value_range", None)
    if cached is not None:
        return cached
    low, high = np.inf, -np.inf
    for data in collection.block_data.values():
        values = np.asarray(data)
        low = min(low, float(np.min(values, where=values > 0, initial=np.inf)))
        high = max(high, float(values.max(initial=-np.inf)))
    cached = (low, high) if np.isfinite(low) and high > 0 else None
    collection._yt_volumes_value_range = cached
    return cached


def _value_distribution(collection):
    """Histograms of log10 of the positive texel values, weighted by cell
    volume and by mass (physical value times volume): `(bin edges, volume
    weights, mass weights)`, cached on the collection; None if no value is
    positive.
    """
    if hasattr(collection, "_yt_volumes_value_distribution"):
        return collection._yt_volumes_value_distribution
    value_range = _positive_value_range(collection)
    distribution = None
    if value_range is not None:
        edges = np.linspace(np.log10(value_range[0]), np.log10(value_range[1]), _DISTRIBUTION_BINS + 1)
        widths = _vertex_attribute(collection, "in_dx")
        volume, mass = np.zeros(_DISTRIBUTION_BINS), np.zeros(_DISTRIBUTION_BINS)
        for vbo_i, data in collection.block_data.items():
            values = np.asarray(data)
            values = values[values > 0]
            if values.size:
                cell_volume = float(np.prod(widths[vbo_i]))
                log_values = np.log10(values)
                volume += np.histogram(log_values, bins=edges)[0] * cell_volume
                mass += np.histogram(log_values, bins=edges, weights=_physical(collection, values))[0] * cell_volume
        distribution = (edges, volume, mass)
    collection._yt_volumes_value_distribution = distribution
    return distribution


def _cumulative(edges, weights):
    return np.concatenate([[0.0], np.cumsum(weights)]) / weights.sum()


def _percentile_range(distribution, low, high):
    """Normalized values at the `low` and `high` percentiles (by volume)."""
    edges, volume, _ = distribution
    cdf = _cumulative(edges, volume)
    # The densest cells' volumes can vanish in the cumulative sum, so the ends
    # come from the outermost occupied bins rather than where the sum reaches 0 or 1.
    occupied = np.nonzero(volume)[0]

    def at(percentile):
        if percentile <= 0.0:
            return edges[occupied[0]]
        if percentile >= 100.0:
            return edges[occupied[-1] + 1]
        return np.interp(percentile / 100.0, cdf, edges)

    return 10.0 ** at(low), 10.0 ** at(high)


def _point_distribution(collection, distribution, value_min, value_max, n_points):
    """Per transfer-function point, over [value_min, value_max] in log space:
    the fraction of the volume and of the mass in its bin (point i covers x
    within half a spacing of i / (n_points - 1)), and log10 of the physical
    value at the point.
    """
    edges, volume, mass = distribution
    low, high = np.log10(value_min), np.log10(value_max)
    centers = np.linspace(low, high, n_points)
    half = (high - low) / (n_points - 1) / 2
    bounds = np.concatenate([[centers[0] - half], centers + half])
    fractions = [np.diff(np.interp(bounds, edges, _cumulative(edges, w))) for w in (volume, mass)]
    log_value = np.log10(np.maximum(_physical(collection, 10.0 ** centers), np.finfo(np.float64).tiny))
    return {"volume_fraction": fractions[0], "mass_fraction": fractions[1], "log10_value": log_value}


def _normalized(collection, value):
    """A value in the field's own units as yt_idv's texels hold it; floored
    at float32's smallest normal, since the shader takes its log.
    """
    if collection._textures_are_normalized:
        value = (value - collection.min_val) / (collection.max_val - collection.min_val)
    return max(value, float(np.finfo(np.float32).tiny))


def _physical(collection, value):
    """A normalized texel value in the field's own units."""
    if collection._textures_are_normalized:
        return value * (collection.max_val - collection.min_val) + collection.min_val
    return value


def layer_bounds(dataset_id):
    """The log10 `(low, high)` the yt Layers mode last used for a loaded
    dataset, or None.
    """
    cached = _volume_scene_cache.get(dataset_id)
    collections = [] if cached is None else _block_collections(cached[0])
    return getattr(collections[0], "_yt_volumes_layer_bounds", None) if collections else None


def gn_tf_fit(dataset_id):
    """The last Geometry Nodes transfer function fit for a loaded dataset:
    `{"physical_range", "units", "points"}` (see `_point_distribution`), or None.
    """
    cached = _volume_scene_cache.get(dataset_id)
    collections = [] if cached is None else _block_collections(cached[0])
    return getattr(collections[0], "_yt_volumes_gn_tf_fit", None) if collections else None


def _configure_gn_transfer_function(component, custom_params):
    """Geometry Nodes transfer function: `yt_idv_gn_tf_rgba` (rows of sRGB
    color and alpha in [0, 1]) over the data's range between
    `yt_idv_gn_tf_percentile_low`/`_high` (by volume), or without
    `yt_idv_gn_tf_fit_range` over `yt_idv_gn_tf_value_min`/`_max`: log10 of
    the field in its own units.
    """
    rgba = custom_params.get("yt_idv_gn_tf_rgba")
    if rgba is None:
        raise ValueError("the Geometry Nodes transfer function wasn't sampled (no yt_idv_gn_tf_rgba)")
    component.render_method = "transfer_function"
    scale = float(custom_params.get("yt_idv_gn_tf_opacity_scale", 1.0))
    data = (np.clip(np.asarray(rgba, dtype=np.float64), 0.0, 1.0) * scale).astype("f4").reshape(len(rgba), 1, 4)
    value_min, value_max = (
        _normalized(component.data, 10.0 ** float(custom_params.get(f"yt_idv_gn_tf_value_{end}", default)))
        for end, default in (("min", -28.0), ("max", -10.0))
    )
    distribution = _value_distribution(component.data)
    if distribution is not None and custom_params.get("yt_idv_gn_tf_fit_range", True):
        value_min, value_max = _percentile_range(
            distribution,
            custom_params.get("yt_idv_gn_tf_percentile_low", 1.0),
            custom_params.get("yt_idv_gn_tf_percentile_high", 100.0),
        )
        # The shader drops samples outside [tf_min, tf_max]; float32 texels can round past float64 ends.
        value_min, value_max = value_min * (1 - 1e-6), value_max * (1 + 1e-6)
    if distribution is not None:
        component.data._yt_volumes_gn_tf_fit = {
            "physical_range": (_physical(component.data, value_min), _physical(component.data, value_max)),
            "units": getattr(component.data, "field_units", None) or "",
            "points": _point_distribution(component.data, distribution, value_min, value_max, len(rgba)),
        }
    # The default table is 8-bit, which clamps to 1; a texture's storage is fixed by
    # its first upload, so each component keeps one float32 table, updated in place.
    texture = _gn_tf_textures.get(id(component))
    if texture is None or texture.data.shape != data.shape:
        from yt_idv.opengl_support import TransferFunctionTexture

        texture = _gn_tf_textures[id(component)] = TransferFunctionTexture(data=data)
    component.transfer_function = texture
    _apply_transfer_function_settings(component, value_min, value_max, data)


def yt_idv_gn_transfer_function_renderer(request: RenderRequest) -> RenderResult:
    """`RENDERER_MODES["YT_IDV_GN_TRANSFER_FUNCTION"]`: a transfer function
    sampled from the volume object's Geometry Nodes.
    """
    return _render_with_yt_idv(request, _configure_gn_transfer_function)


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


def _field_value_at(scene, point):
    """The field at unitary `point`, from the block textures (nearest texel),
    in the field's own units; None outside every block.
    """
    for collection in _block_collections(scene):
        if _is_spherical(collection) or not hasattr(collection, "_get_unscaled_edges"):
            continue
        left, right, dx = collection._get_unscaled_edges()
        inside = np.nonzero(np.all((left <= point) & (point < right), axis=1))[0]
        if inside.size == 0:
            continue
        block = inside[np.argmin(dx[inside].max(axis=1))]  # the finest, if blocks overlap
        data = np.asarray(collection.block_data[block])
        # textures are vertex-centered: texel i sits at left + i * dx
        index = np.clip(np.round((point - left[block]) / dx[block]).astype(int), 0, np.array(data.shape) - 1)
        value = float(data[tuple(index)])
        if collection._textures_are_normalized:
            value = value * collection.val_range + collection.min_val
        return value
    return None


def _configure_yt_transfer_function(component, custom_params):
    """A yt ColorTransferFunction from `yt_idv_yt_tf`, converted for this
    component's blocks with yt_idv.tf_conversion (yt-project/yt_idv#288).

    A function taking no arguments is converted once per set of blocks and
    settings. One taking an argument gets a context dict every frame: `frame`,
    `camera` (unitary float64 position), `value` (the field there, from the
    blocks; None outside them) and `ds`.
    """
    spec = custom_params.get("yt_idv_yt_tf")
    if not spec:
        raise ValueError("the yt Transfer Function mode needs yt_idv_yt_tf (Preferences > yt Transfer Function)")
    make = _resolve_callable(spec)
    if inspect.signature(make).parameters:
        _apply_yt_transfer_function(component, custom_params, None, lambda: make(custom_params.get("_context", {})))
    else:
        _apply_yt_transfer_function(component, custom_params, spec, make)


def _apply_yt_transfer_function(component, custom_params, key, make):
    """Convert and apply the yt ColorTransferFunction `make()` returns, with
    `yt_idv_yt_tf_ray_length` and `_sample_factor`; reused while `key` (None:
    never) and the blocks are unchanged.
    """
    from yt_idv.tf_conversion import apply_converted_transfer_function, convert_yt_transfer_function

    sample_factor = float(custom_params.get("yt_idv_yt_tf_sample_factor", 0.0))
    if sample_factor > 0:
        component.sample_factor = sample_factor
    ray_length = float(custom_params.get("yt_idv_yt_tf_ray_length", 0.02))
    per_frame = key is None
    key = (id(component.data), key, ray_length)
    applied = _yt_tf_applied.get(id(component))
    if applied is not None and applied[0] == key and not per_frame:
        if component.transfer_function is applied[1]["transfer_function"]:
            return
        # another mode attached its own table in between
        apply_converted_transfer_function(component, applied[1])
        return
    tf = make()
    try:
        # thin yt layers need a finer table than the 1024-texel default
        converted = convert_yt_transfer_function(tf, component.data, ray_length, nbins=4096)
    except ValueError as exc:
        if "zero everywhere" not in str(exc):
            raise
        # nothing to draw: a range above every normalized value discards all samples
        component.render_method = "transfer_function"
        component.tf_min, component.tf_max = 2.0, 3.0
        _yt_tf_applied.pop(id(component), None)
        return
    apply_converted_transfer_function(component, converted)
    if applied is not None:
        applied[1]["transfer_function"].release()
    _yt_tf_applied[id(component)] = (key, converted)


def yt_idv_yt_transfer_function_renderer(request: RenderRequest) -> RenderResult:
    """`RENDERER_MODES["YT_IDV_YT_TRANSFER_FUNCTION"]`: a yt-built transfer function."""
    return _render_with_yt_idv(request, _configure_yt_transfer_function)


def _configure_projection(component, custom_params):
    component.render_method = "projection"
    _set_colormap_name(component, custom_params)


def yt_idv_projection_renderer(request: RenderRequest) -> RenderResult:
    """`RENDERER_MODES["YT_IDV_PROJECTION"]`: additive line-of-sight
    integration.
    """
    return _render_with_yt_idv(request, _configure_projection)
