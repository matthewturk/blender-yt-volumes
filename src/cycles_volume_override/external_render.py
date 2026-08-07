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
class VolumeObjectParams:
    matrix_world: list  # 4x4 nested list of float, object-to-world
    bounding_box_local: list  # 8 corners, object-local space
    custom_params: dict  # passthrough from the PropertyGroup:
    #                       density scale, absorption/scattering coeffs,
    #                       VDB/grid file path, color, etc.


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
