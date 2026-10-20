"""Interface between this add-on and the external volume renderer.

A renderer is a function `RenderRequest -> RenderResult`, one per renderer
mode (`operators.RENDERER_MODES`). Prefer adding renderer-specific inputs to
`VolumeObjectParams.custom_params` over growing these dataclasses.

The renderer is self-emissive: no lighting or shadow data is passed.
"""

from dataclasses import dataclass, field


@dataclass
class PanoramicLensParams:
    """Raw Cycles panorama fields; set only for PANO cameras.

    Verified for EQUIRECTANGULAR against Cycles
    (scripts/verify_pano_equirect.py), in camera-local space (forward = -Z,
    up = +Y, right = +X):

        theta = atan2(x, -z)               # longitude
        u = (theta / (2*pi) + 0.5) % 1.0   # u=0.5 at forward
        phi = asin(y / length((x, y, z)))  # latitude
        v_from_top = 0.5 - phi / pi        # v=0 at up (+Y)

    and inversely:

        theta = (u - 0.5) * 2*pi
        phi = (0.5 - v_from_top) * pi
        direction = (sin(theta)*cos(phi), sin(phi), -cos(theta)*cos(phi))

    The fisheye, mirrorball, and cylindrical sub-types are not verified; see
    Cycles' `kernel/camera/projection.h`.
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
    """Lens/aperture settings for depth of field.

    `focus_distance_m` is always a float: when a focus object is set, it is
    the Euclidean camera-to-object distance (not verified against Cycles).
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
    # 4x4 camera-to-clip matrix from `Object.calc_matrix_camera()` at the
    # render resolution/pixel aspect; with `matrix_world.inverted()` it maps
    # pixel + depth <-> world point. None for PANO (use `panorama`).
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
    """Content hash plus a Blender-side "changed since last request" flag.

    Used separately for `custom_params` (cheap to re-apply) and for the
    dataset (expensive to reload).

    `hash` is stable across restarts and transports; cache loaded data keyed
    by (`DatasetParams.dataset_id`, `hash`).

    `changed` compares against the last hash Blender sent for this object.
    It resets to True after an add-on reload or Blender restart and says
    nothing about the renderer's own cache, so `changed=False` means nothing
    to do but `changed=True` should be confirmed against `hash`.
    """

    hash: str
    changed: bool


@dataclass
class DatasetParams:
    """Identity of the dataset this volume object represents (format-agnostic).

    `dataset_path`/`dataset_kind` come from the object's properties, except
    for Blender `Volume` objects, where they are read from `obj.data`.
    Loader-specific arguments belong in `VolumeObjectParams.custom_params`.
    """

    dataset_id: str  # stable UUID, survives renames
    dataset_kind: str  # free-form: "yt" | "vdb" | "hdf5" | "" (unset) | anything your loaders recognize
    dataset_path: str | None  # absolute path/URI; None if unset

    # Set only for Blender `Volume` objects (VDB sequences); None/False
    # otherwise. `resolved_sequence_frame` follows Blender's documented
    # sequence semantics but is not verified against a real render.
    is_blender_volume_sequence: bool
    sequence_frame_start: int | None
    sequence_frame_duration: int | None
    sequence_frame_offset: int | None
    sequence_mode: str | None  # "CLIP" | "EXTEND" | "REPEAT" | "PING_PONG"
    resolved_sequence_frame: int | None  # best-effort
    grid_names: list  # metadata only (names Blender reports for the loaded frame), NOT voxel data

    version: VersionInfo  # use .hash as the cache key


@dataclass
class VolumeObjectParams:
    matrix_world: list  # 4x4 nested list of float, object-to-world
    bounding_box_local: list  # 8 corners, object-local space
    custom_params: dict  # shading params from the PropertyGroup
    params_version: VersionInfo  # change tracking for custom_params specifically
    dataset: DatasetParams  # the (possibly large) dataset this object represents


@dataclass
class FrameParams:
    frame_number: int
    scene_time_seconds: float
    view_index: int = 0  # stereo: 0 for the first eye rendered, 1 for the second


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
    cmap_range: tuple | None = None  # (min, max) the colormap used, remembered on the volume
