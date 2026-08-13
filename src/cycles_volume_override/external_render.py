"""Hard boundary between this add-on and the external volume renderer.

Nothing in this add-on talks to the real renderer except through
`render_volume()`. The dataclass shapes below are the *frozen* interface
contract (see prompt.md) -- do not add fields casually; if the real renderer
needs something new, add it to `VolumeObjectParams.custom_params` first
(it's a passthrough dict) rather than growing the dataclasses, unless it's
something every volume object needs structurally (like the depth buffer).

No lighting/shadow data crosses this boundary in v1: the renderer is fully
self-emissive and does not need Blender's light list, shadow maps, or
irradiance. A per-light shadow-map-style depth pass is the documented future
extension point if we ever want Cycles-cast light on the volume -- see
README "Future work".

Transport: this is an in-process Python call today. Keeping ALL of that
boundary behind `render_volume()` (rather than, say, having callers reach
into a subprocess or socket themselves) is what makes swapping to a
subprocess or network call later a one-function change -- see README
"Swappable transport".

Dataset identity + change tracking (`DatasetParams`, `VersionInfo` below)
crosses this boundary too, separately from the per-frame shading knobs in
`custom_params` -- see change_tracking.py and README "Dataset / change
tracking" for why they're split and how to use the hash vs. the convenience
`changed` flag.
"""

from dataclasses import dataclass, field


@dataclass
class PanoramicLensParams:
    """PANO only (`CameraParams.panorama` is None otherwise). Raw Cycles
    panorama fields, passed through verbatim rather than pre-converted to a
    ray function on the Blender side -- so your renderer can implement
    Cycles' own published per-type ray-generation formula directly, matching
    Cycles pixel-for-pixel for whichever sub-type you actually use.

    Empirically confirmed here (scripts/verify_pano_equirect.py, marker
    round-trip against real Cycles render): for `panorama_type ==
    "EQUIRECTANGULAR"`, in **camera-local space** (forward = -Z, up = +Y,
    right = +X, matching `CameraParams.matrix_world`'s convention):

        theta = atan2(x, -z)              # longitude
        u = (theta / (2*pi) + 0.5) % 1.0   # u=0.5 at forward, 0.75 right, 0.25 left, 0/1 back
        phi = asin(y / length((x, y, z)))  # latitude
        v_from_top = 0.5 - phi / pi        # v=0 at up (+Y), v=1 at down (-Y)

    which is the standard equirectangular map -- cleanly invertible except at
    the poles (longitude is genuinely undefined there, same as ordinary
    spherical coordinates):

        theta = (u - 0.5) * 2*pi
        phi = (0.5 - v_from_top) * pi
        direction = (sin(theta)*cos(phi), sin(phi), -cos(theta)*cos(phi))

    The other sub-types (`FISHEYE_EQUIDISTANT`, `FISHEYE_EQUISOLID`,
    `FISHEYE_LENS_POLYNOMIAL`, `MIRRORBALL`, `CENTRAL_CYLINDRICAL`,
    `EQUIANGULAR_CUBEMAP_FACE`) are NOT independently verified against a real
    Cycles render the way equirectangular was -- their formulas below follow
    Cycles' publicly documented camera model but haven't been round-trip
    checked here. Re-verify with the same marker technique
    (scripts/verify_pano_equirect.py) before trusting one of those in
    production. Standard formulas, camera-local space, `r = fisheye_lens_mm`
    (or per-sub-type field as noted):

        FISHEYE_EQUIDISTANT: r_px = fisheye_lens_mm * theta_from_forward
        FISHEYE_EQUISOLID:   r_px = 2 * fisheye_lens_mm * sin(theta_from_forward / 2)

    where `theta_from_forward` is the angle between the ray and the camera's
    forward (-Z) axis, and `r_px` is the radial pixel distance from the
    image center, scaled by `fisheye_fov_rad` to fit the frame. Consult
    Cycles' own source (`kernel/camera/projection.h`) for
    `FISHEYE_LENS_POLYNOMIAL` (a 4th-order polynomial in `r_px`, coefficients
    `fisheye_polynomial_k0..k4`) and `MIRRORBALL`/`CENTRAL_CYLINDRICAL` before
    relying on the sketch above.
    """

    panorama_type: str  # "EQUIRECTANGULAR" | "FISHEYE_EQUIDISTANT" | "FISHEYE_EQUISOLID" | "FISHEYE_LENS_POLYNOMIAL" | "MIRRORBALL" | "CENTRAL_CYLINDRICAL" | "EQUIANGULAR_CUBEMAP_FACE"
    fisheye_lens_mm: float
    fisheye_fov_rad: float
    fisheye_polynomial_k0: float
    fisheye_polynomial_k1: float
    fisheye_polynomial_k2: float
    fisheye_polynomial_k3: float
    fisheye_polynomial_k4: float
    latitude_min_rad: float
    latitude_max_rad: float
    longitude_min_rad: float
    longitude_max_rad: float
    central_cylindrical_radius: float
    central_cylindrical_range_u_min: float
    central_cylindrical_range_u_max: float
    central_cylindrical_range_v_min: float
    central_cylindrical_range_v_max: float


@dataclass
class DepthOfFieldParams:
    """Lens/aperture info for physically-based path tracing (bokeh, focus
    blur) -- not part of the original contract, added because the renderer
    needs to replicate Cycles' lens behavior, not just its camera geometry.

    `focus_distance_m` is resolved ON THE BLENDER SIDE to a plain float even
    when the camera's DOF is target-object-driven (`dof.focus_object` set):
    computed as Euclidean world-space distance from the camera to that
    object/bone, since `camera.data.dof.focus_distance` itself is stale
    (unused) whenever a focus object is set. **Caveat, same category as the
    Z-pass Euclidean-vs-planar question**: this Euclidean-distance choice has
    NOT been empirically checked against what Cycles' own DOF math actually
    uses internally for a focus object -- if precise DOF matching matters,
    verify with the same marker technique used elsewhere in this repo before
    trusting it.
    """

    use_dof: bool
    focus_distance_m: float
    aperture_fstop: float
    aperture_blades: int
    aperture_ratio: float
    aperture_rotation_rad: float


@dataclass
class CameraParams:
    matrix_world: list  # 4x4 nested list of float, camera-to-world
    projection: str  # "PERSP" | "ORTHO" | "PANO"
    lens_mm: float | None  # perspective only
    sensor_width_mm: float
    sensor_height_mm: float
    sensor_fit: str  # "AUTO" | "HORIZONTAL" | "VERTICAL"
    shift_x: float
    shift_y: float
    ortho_scale: float | None  # ortho only
    clip_start: float
    clip_end: float
    resolution_x: int
    resolution_y: int
    pixel_aspect_x: float
    pixel_aspect_y: float
    projection_matrix: list | None
    # 4x4 camera-to-clip-space matrix from Blender's own
    # `Object.calc_matrix_camera(depsgraph, x=resolution_x, y=resolution_y,
    # scale_x=pixel_aspect_x, scale_y=pixel_aspect_y)`. Combined with
    # `matrix_world.inverted()`, gives one linear, invertible transform:
    # `pixel + depth -> unique 3D world point` and back. PERSP/ORTHO only --
    # None for PANO, which has no single-matrix representation (use
    # `panorama` instead). Empirically round-trip verified against real
    # Cycles renders for both PERSP and ORTHO to sub-pixel accuracy; see
    # scripts/verify_camera_projection.py and README "Camera projection".
    panorama: "PanoramicLensParams | None"  # PANO only, else None
    dof: "DepthOfFieldParams"


@dataclass
class DepthBuffer:
    data: "numpy.ndarray"  # shape (resolution_y, resolution_x), float32
    depth_convention: str  # "EUCLIDEAN" | "PLANAR" -- see scripts/verify_zpass.py
    no_hit_value: float  # sentinel for background/sky pixels; must NOT
    #                       be treated as a real occluder
    row_order: str  # "TOP_LEFT_ORIGIN" | "BOTTOM_LEFT_ORIGIN"
    units: str  # e.g. "meters" -- scene unit scale


@dataclass
class VersionInfo:
    """A content hash + a Blender-computed "changed since I last built a
    request for this object" convenience flag. Used for BOTH
    `VolumeObjectParams.params_version` (custom_params) and
    `DatasetParams.version` (the dataset itself) -- two independent signals,
    because they have very different costs on the renderer's side: shading
    knobs are cheap to re-apply every frame, a multi-GB scientific dataset is
    not something you want to reload/reparse unless it actually changed.

    `hash` is the transport-safe source of truth for caching: it's a plain
    content hash (see change_tracking.py), stable across process restarts
    and independent of Blender's own bookkeeping. Cache your loaded/parsed
    data keyed by (`DatasetParams.dataset_id`, `hash`); if the incoming hash
    matches what you have cached, reuse it -- this works correctly even if
    your renderer process restarted, the transport changed (in-process ->
    subprocess -> network), or Blender's own change-tracking state was
    cleared for any reason. THIS is what "did it change" should really mean.

    `changed` is a convenience on top of that, computed by Blender comparing
    this call's hash to the last hash IT sent for this object
    (change_tracking.py's in-memory record). It is NOT authoritative:
      - Resets on add-on reload / Blender restart -- the first call after
        either always reports `changed=True` regardless of whether anything
        actually changed, because Blender has no memory of what it last sent.
      - Only reflects whether BLENDER's inputs changed -- says nothing about
        whether your OWN cache is still warm (e.g. your subprocess restarted
        independently of Blender).
    Treat `changed=False` as a reliable fast-path ("definitely nothing to
    do"); do NOT treat `changed=True` as proof you need to redo work --
    always fall back to comparing `hash` against your own cache to be sure.
    """

    hash: str
    changed: bool


@dataclass
class DatasetParams:
    """Identity of the actual (potentially large/expensive-to-load) dataset
    this volume object represents -- deliberately generic, NOT tied to any
    one file format. The end goal is arbitrary scientific datasets (e.g. a
    yt-loadable simulation output), with Blender-native VDB `Volume` objects
    as just one possible source Blender happens to be able to introspect
    automatically -- see `dataset_kind`.

    `dataset_path`/`dataset_kind` mirror `properties.py`'s
    `dataset_path`/`dataset_kind` fields directly (whatever the user typed
    there) UNLESS the flagged object is a Blender `Volume` object, in which
    case `operators.build_dataset_params()` overrides them with values read
    from `obj.data` (native VDB filepath/sequence info) instead -- Blender
    already knows that path itself, no need to duplicate it by hand. Either
    way, any loader-specific arguments beyond a path (a yt field name, an
    HDF5 dataset key, a unit system) belong in
    `VolumeObjectParams.custom_params`, not here.
    """

    dataset_id: str  # stable cache key -- properties.ensure_dataset_uuid(obj), NOT obj.name (survives renames)
    dataset_kind: str  # free-form: "yt" | "vdb" | "hdf5" | "" (unset) | anything your loaders recognize
    dataset_path: str | None  # absolute path/URI; None if unset

    # --- Blender frame -> dataset time mapping ---
    # `requested_time` is Blender's resolved answer to "what point in the
    # dataset's own timeline does the current frame correspond to" --
    # deliberately NOT snapped to an actual timestep/snapshot. Blender
    # doesn't open the dataset, so it has no way to know what timesteps
    # actually exist; snapping `requested_time` to the nearest real one is
    # the renderer's job. `mapping_policy` records which
    # properties.py policy produced it (DIRECT/AFFINE/MANUAL -- see
    # operators.resolve_requested_time()), for introspection/debugging, not
    # something the renderer needs to branch on. `time_units` says what the
    # number actually means (a plain frame index vs. seconds vs. redshift
    # vs. something dataset-specific) -- same "be explicit, don't assume"
    # discipline as `DepthBuffer.units`.
    #
    # Deliberately EXCLUDED from `version`'s hash below, even though it's
    # right here next to dataset_path/dataset_kind: it changes essentially
    # every frame by design (that's the whole point), so hashing it in would
    # make `version.changed` report True every single frame regardless of
    # whether the SNAPPED result actually differs -- exactly the
    # over-invalidation `params_version`/`dataset.version`'s split was meant
    # to avoid in the first place. The renderer, which knows the real
    # timesteps, is better positioned to decide for itself whether two
    # different `requested_time` values snap to the same timestep and skip
    # reloading -- that decision needs dataset knowledge Blender doesn't have.
    requested_time: float
    time_units: str  # "FRAME_INDEX" | "SECONDS" | "REDSHIFT" | "CUSTOM"
    mapping_policy: str  # "DIRECT" | "AFFINE" | "MANUAL" -- see properties.py

    # Populated only when the flagged object is a Blender `Volume` object
    # (obj.type == "VOLUME") wrapping a VDB (sequence); None/False otherwise.
    # NOT independently empirically verified the way the Z-pass/camera
    # projection findings were -- `resolved_frame` in particular replicates
    # Blender's documented start/duration/offset/clip-extend-repeat-pingpong
    # semantics from reading the docs, not from a round-trip render check.
    # Re-verify against a real VDB sequence before trusting it exactly.
    is_blender_volume_sequence: bool
    sequence_frame_start: int | None
    sequence_frame_duration: int | None
    sequence_frame_offset: int | None
    sequence_mode: str | None  # "CLIP" | "EXTEND" | "REPEAT" | "PING_PONG"
    resolved_sequence_frame: int | None  # best-effort; see caveat above
    grid_names: list  # metadata only (names Blender reports for the loaded frame), NOT voxel data

    version: VersionInfo  # see VersionInfo docstring -- use .hash as your cache key, not .changed alone


@dataclass
class VolumeObjectParams:
    matrix_world: list  # 4x4 nested list of float, object-to-world
    bounding_box_local: list  # 8 corners, object-local space
    custom_params: dict  # passthrough from the PropertyGroup:
    #                       density scale, absorption/scattering coeffs,
    #                       color, extra key/value pairs, etc. -- NOT
    #                       dataset identity, see `dataset` below.
    params_version: VersionInfo  # change tracking for custom_params specifically
    dataset: DatasetParams  # the (possibly large) dataset this object represents


@dataclass
class FrameParams:
    frame_number: int
    scene_time_seconds: float


@dataclass
class RenderRequest:
    camera: CameraParams
    depth: DepthBuffer
    volume: VolumeObjectParams
    frame: FrameParams


@dataclass
class RenderResult:
    rgba: "numpy.ndarray"  # shape (resolution_y, resolution_x, 4), float32
    alpha_mode: str  # "PREMULTIPLIED" -- fixed, must match compositor
    color_space: str  # "LINEAR" -- fixed, must match compositor
    #                    (no view transform applied by the renderer)


def render_volume(request: RenderRequest) -> RenderResult:
    """Stub -- calls out to the real external volume renderer.

    Wire the real call in here. Do not change the signature: `operators.py`
    builds exactly one `RenderRequest` per flagged volume object per frame
    and expects exactly one `RenderResult` back, synchronously. If the real
    renderer is async/out-of-process, block on it here (or raise and let the
    render fail loudly) -- callers are not written to handle a partial or
    deferred result.
    """
    raise NotImplementedError(
        "render_volume() is a stub -- wire in the real external volume "
        "renderer here. See README 'external_render.py contract'."
    )


def debug_fake_render_volume(request: RenderRequest) -> RenderResult:
    """Non-contract debug stand-in, NOT called anywhere by default.

    Exists only so the rest of the pipeline (depth read-back, compositor
    graph, two-pass render orchestration) can be exercised end-to-end before
    the real renderer is wired in. Enable it via the add-on preference
    "Use debug fake volume renderer" (see operators.py) -- never enabled
    silently. Produces a flat-colored, semi-transparent premultiplied RGBA
    filling the volume object's projected bounding box, clipped at the
    supplied depth buffer, so you can see the two-pass compositing and
    depth-clip machinery actually working without the real renderer.
    """
    import numpy as np

    h, w = request.depth.data.shape
    rgba = np.zeros((h, w, 4), dtype=np.float32)
    color = request.volume.custom_params.get("color", (0.4, 0.6, 1.0))
    alpha = 0.25
    rgba[..., 0] = color[0] * alpha
    rgba[..., 1] = color[1] * alpha
    rgba[..., 2] = color[2] * alpha
    rgba[..., 3] = alpha
    return RenderResult(rgba=rgba, alpha_mode="PREMULTIPLIED", color_space="LINEAR")
