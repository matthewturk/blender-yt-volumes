"""Minimal, non-physical `render_volume()` for testing the pipeline:

    output = (ray path length through the volume's box, clipped at the
              solid surface behind it) * a constant

Pure numpy, no `bpy`. Selected by the `SAMPLE_CONSTANT` renderer mode.
Supports PERSP/ORTHO and EQUIRECTANGULAR PANO cameras.
"""

import numpy as np

from .external_render import RenderRequest, RenderResult


def constant_density_renderer(request: RenderRequest) -> RenderResult:
    camera = request.camera
    depth = request.depth
    volume = request.volume

    if depth.depth_convention != "PLANAR":
        # Cycles' Z pass is planar depth.
        raise NotImplementedError(
            "constant_density_renderer assumes DepthBuffer.depth_convention "
            f"== 'PLANAR'; got {depth.depth_convention!r}."
        )

    origins, directions = _pixel_rays(camera, depth.row_order, depth.data.shape)
    box_min, box_max = _world_aabb(volume.matrix_world, volume.bounding_box_local)
    t_near, t_far, hit = _ray_aabb_intersect(origins, directions, box_min, box_max)

    matrix_world_inv = np.linalg.inv(np.array(camera.matrix_world, dtype=np.float64))
    p_near = origins + t_near[..., None] * directions
    p_far = origins + t_far[..., None] * directions
    depth_near = _planar_depth(p_near, matrix_world_inv)
    depth_far = _planar_depth(p_far, matrix_world_inv)

    # no_hit_value (~1e10) never clips, so no special case is needed.
    clipped_far = np.minimum(depth_far, depth.data)
    path_length = np.maximum(clipped_far - depth_near, 0.0)
    path_length = np.where(hit, path_length, 0.0)

    density_scale = float(volume.custom_params.get("density_scale", 1.0))
    color = np.asarray(volume.custom_params.get("color", (0.4, 0.6, 1.0)), dtype=np.float32)

    alpha = np.clip(path_length * density_scale, 0.0, 1.0).astype(np.float32)
    rgba = np.zeros((*alpha.shape, 4), dtype=np.float32)
    rgba[..., :3] = color[None, None, :] * alpha[..., None]  # premultiplied
    rgba[..., 3] = alpha

    return RenderResult(rgba=rgba, alpha_mode="PREMULTIPLIED", color_space="LINEAR")


# ---------------------------------------------------------------------------
# Ray generation
# ---------------------------------------------------------------------------

def _pixel_grid_fractions(row_order, shape):
    """Pixel-center fractions (col_frac, row_frac_from_bottom) in [0, 1];
    `row_frac_from_bottom` is 0 at the bottom row whatever `row_order` is.
    """
    h, w = shape
    col_frac = (np.arange(w) + 0.5) / w
    row_idx = np.arange(h)
    if row_order == "BOTTOM_LEFT_ORIGIN":
        row_frac_from_bottom = (row_idx + 0.5) / h
    elif row_order == "TOP_LEFT_ORIGIN":
        row_frac_from_bottom = 1.0 - (row_idx + 0.5) / h
    else:
        raise ValueError(f"unknown row_order {row_order!r}")
    return col_frac, row_frac_from_bottom


def _pixel_rays(camera, row_order, shape):
    """(origins, directions), each (H, W, 3) world space; directions are not
    normalized.
    """
    if camera.projection in ("PERSP", "ORTHO"):
        return _persp_ortho_rays(camera, row_order, shape)
    if camera.projection == "PANO":
        if camera.panorama is None or camera.panorama.panorama_type != "EQUIRECTANGULAR":
            got = camera.panorama.panorama_type if camera.panorama else None
            raise NotImplementedError(
                "constant_density_renderer only supports PANO cameras with "
                f"panorama_type == 'EQUIRECTANGULAR'; got {got!r}. See "
                "PanoramicLensParams' docstring for the other sub-types' "
                "(unverified) formulas if you need one of those."
            )
        return _equirect_rays(camera, row_order, shape)
    raise NotImplementedError(f"unknown CameraParams.projection {camera.projection!r}")


def _persp_ortho_rays(camera, row_order, shape):
    if camera.projection_matrix is None:
        raise ValueError("PERSP/ORTHO camera with no projection_matrix -- see operators.build_camera_params()")

    h, w = shape
    col_frac, row_frac_from_bottom = _pixel_grid_fractions(row_order, shape)
    ndc_x = col_frac * 2.0 - 1.0
    ndc_y = row_frac_from_bottom * 2.0 - 1.0
    ndc_x_grid, ndc_y_grid = np.meshgrid(ndc_x, ndc_y)  # each (h, w)

    proj_inv = np.linalg.inv(np.array(camera.projection_matrix, dtype=np.float64))
    matrix_world = np.array(camera.matrix_world, dtype=np.float64)

    def unproject(ndc_z):
        clip = np.stack(
            [ndc_x_grid, ndc_y_grid, np.full((h, w), ndc_z), np.ones((h, w))], axis=-1,
        )
        cam_space = clip @ proj_inv.T
        cam_space = cam_space[..., :3] / cam_space[..., 3:4]
        world_h = np.concatenate([cam_space, np.ones((h, w, 1))], axis=-1)
        world = world_h @ matrix_world.T
        return world[..., :3]

    # Any two distinct ndc_z planes define the pixel's ray.
    p_a = unproject(0.0)
    p_b = unproject(1.0)
    return p_a, (p_b - p_a)


def _equirect_rays(camera, row_order, shape):
    """Equirectangular rays; camera-local forward = -Z, up = +Y, right = +X."""
    h, w = shape
    col_frac, row_frac_from_bottom = _pixel_grid_fractions(row_order, shape)
    u_grid, v_from_bottom_grid = np.meshgrid(col_frac, row_frac_from_bottom)
    v_from_top_grid = 1.0 - v_from_bottom_grid

    theta = (u_grid - 0.5) * 2.0 * np.pi
    phi = (0.5 - v_from_top_grid) * np.pi
    dir_cam = np.stack(
        [np.sin(theta) * np.cos(phi), np.sin(phi), -np.cos(theta) * np.cos(phi)], axis=-1,
    )

    matrix_world = np.array(camera.matrix_world, dtype=np.float64)
    rotation = matrix_world[:3, :3]
    dir_world = dir_cam @ rotation.T
    origin_world = np.broadcast_to(matrix_world[:3, 3], dir_world.shape)
    return origin_world.copy(), dir_world


# ---------------------------------------------------------------------------
# Volume bounding box + ray/box intersection
# ---------------------------------------------------------------------------

def _world_aabb(matrix_world_list, bbox_local):
    """World-space AABB of the object's 8 local bounding-box corners."""
    matrix_world = np.array(matrix_world_list, dtype=np.float64)
    corners = np.asarray(bbox_local, dtype=np.float64)  # (8, 3)
    corners_h = np.concatenate([corners, np.ones((8, 1))], axis=-1)
    world = corners_h @ matrix_world.T
    world = world[:, :3] / world[:, 3:4]
    return world.min(axis=0), world.max(axis=0)


def _ray_aabb_intersect(origins, directions, box_min, box_max):
    """Vectorized slab test. Returns (t_near, t_far, hit), with t_near
    clamped to >= 0.
    """
    with np.errstate(divide="ignore", invalid="ignore"):
        inv_dir = 1.0 / directions
        t1 = (box_min - origins) * inv_dir
        t2 = (box_max - origins) * inv_dir
    t_small = np.minimum(t1, t2)
    t_big = np.maximum(t1, t2)
    t_near = np.max(t_small, axis=-1)
    t_far = np.min(t_big, axis=-1)
    t_near_clamped = np.maximum(t_near, 0.0)
    hit = (t_far >= t_near_clamped) & (t_far >= 0.0)
    return t_near_clamped, t_far, hit


def _planar_depth(world_points, matrix_world_inv):
    """Planar depth (negative camera-space Z) of `world_points`."""
    h, w = world_points.shape[:2]
    points_h = np.concatenate([world_points, np.ones((h, w, 1))], axis=-1)
    cam_space = points_h @ matrix_world_inv.T
    cam_space = cam_space[..., :3] / cam_space[..., 3:4]
    return -cam_space[..., 2]
