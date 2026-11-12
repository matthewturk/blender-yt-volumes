# Ray directions and origins of Blender's panoramic cameras, ported to numpy
# from Cycles:
#   intern/cycles/kernel/camera/projection.h
#     SPDX-FileCopyrightText: 2009-2010 Sony Pictures Imageworks Inc., et al.
#     SPDX-FileCopyrightText: 2011-2022 Blender Foundation
#     SPDX-License-Identifier: BSD-3-Clause (adapted from Open Shading Language)
#   intern/cycles/kernel/camera/camera.h, intern/cycles/scene/camera.cpp,
#   intern/cycles/blender/camera.cpp
#     SPDX-FileCopyrightText: 2011-2022 Blender Foundation
#     SPDX-License-Identifier: Apache-2.0
# as of the blender-v5.2-release branch.
"""Per-pixel rays of a panoramic camera, as Cycles traces them.

tests/test_pano_rays.py checks every panorama type and the stereo settings
against rays Cycles itself renders.
"""

import numpy as np

_cache = {}  # signature -> {view: (directions, origins)}

# Cycles' panorama space looks down +x with +z up; these map it to Blender's
# camera space (forward -Z, up +Y): Blender = M @ panorama
_PANORAMA_TO_CAMERA = np.array([[0.0, -1.0, 0.0], [0.0, 0.0, 1.0], [-1.0, 0.0, 0.0]])
# the mirror ball looks down -y instead
_MIRRORBALL_TO_CAMERA = np.array([[1.0, 0.0, 0.0], [0.0, 0.0, 1.0], [0.0, 1.0, 0.0]])


def render_size(scene):
    render = scene.render
    return (render.resolution_x * render.resolution_percentage // 100,
            render.resolution_y * render.resolution_percentage // 100)


def uses_spherical_stereo(scene, cam_data):
    """True if Cycles places this panorama's stereo eyes per ray; otherwise
    each eye is the whole camera moved sideways, as for a perspective camera.
    """
    return (scene.render.engine == "CYCLES" and cam_data.type == "PANO"
            and cam_data.stereo.use_spherical_stereo)


def _settings(scene, cam_data):
    src = cam_data.cycles if hasattr(cam_data, "cycles") and hasattr(cam_data.cycles, "panorama_type") else cam_data
    stereo = cam_data.stereo
    return {
        "type": src.panorama_type,
        "fisheye_fov": src.fisheye_fov,
        "fisheye_lens": src.fisheye_lens,
        "latitude": (src.latitude_min, src.latitude_max),
        "longitude": (src.longitude_min, src.longitude_max),
        "polynomial": tuple(getattr(src, f"fisheye_polynomial_k{i}") for i in range(5)),
        "cylindrical": tuple(getattr(src, f"central_cylindrical_{name}", default) for name, default in (
            ("range_u_min", -np.pi), ("range_u_max", np.pi), ("range_v_min", -1.0),
            ("range_v_max", 1.0), ("radius", 1.0))),
        "shift": (cam_data.shift_x, cam_data.shift_y),
        "sensor": (cam_data.sensor_fit, cam_data.sensor_width, cam_data.sensor_height),
        "size": render_size(scene),
        "pixel_aspect": (scene.render.pixel_aspect_x, scene.render.pixel_aspect_y),
        "spherical_stereo": uses_spherical_stereo(scene, cam_data),
        "interocular": stereo.interocular_distance,
        "convergence": None if stereo.convergence_mode == "PARALLEL" else stereo.convergence_distance,
        "pole_merge": (stereo.pole_merge_angle_from, stereo.pole_merge_angle_to) if stereo.use_pole_merge else None,
    }


def ray_tables(scene, camera_obj, views=(None,), shift_x=None):
    """`{view: (directions, origins)}` for `camera_obj`'s panorama at the
    scene's render size, `view` being `"left"`/`"right"` or None. Both are
    `(height, width, 3)` float32 in camera space (forward -Z, up +Y, right
    +X), rows bottom first: unit directions, (0, 0, 0) where a pixel has no
    ray, and ray origins relative to the camera, or None when every ray
    starts at the camera (no spherical stereo). `shift_x` overrides the
    camera's, for an eye of a non-spherical stereo pair.
    """
    settings = _settings(scene, camera_obj.data)
    if shift_x is not None:
        settings["shift"] = (shift_x, settings["shift"][1])
    key = (repr(sorted(settings.items())), tuple(views))
    tables = _cache.get(key)
    if tables is None:
        tables = {view: _rays(settings, view) for view in views}
        _cache.clear()
        _cache[key] = tables
    return tables


def _rays(s, view):
    width, height = s["size"]
    # Cycles' panorama viewplane is [0, 1] x [0, 1] offset by the camera
    # shift; raster rows count from the bottom, sampled at pixel centers.
    u = (np.arange(width) + 0.5) / width + s["shift"][0]
    v = (np.arange(height) + 0.5) / height + s["shift"][1]
    u, v = np.meshgrid(u, v)
    d = _panorama_to_direction(s, u, v)
    has_ray = np.any(d != 0.0, axis=-1)

    origins = None
    if view in ("left", "right") and s["spherical_stereo"]:
        offset = s["interocular"] * (-0.5 if view == "left" else 0.5)
        p, d = _spherical_stereo(s, d, offset)
        origins = np.where(has_ray[..., None], p, 0.0)

    to_camera = _MIRRORBALL_TO_CAMERA if s["type"] == "MIRRORBALL" else _PANORAMA_TO_CAMERA
    d = d @ to_camera.T
    norm = np.linalg.norm(d, axis=-1, keepdims=True)
    d = np.where(has_ray[..., None], d / np.where(norm > 0, norm, 1.0), 0.0)
    if origins is not None:
        origins = np.ascontiguousarray(origins @ to_camera.T, dtype=np.float32)
    return np.ascontiguousarray(d, dtype=np.float32), origins


def _spherical_to_direction(theta, phi):
    return np.stack([np.sin(theta) * np.cos(phi), np.sin(theta) * np.sin(phi), np.cos(theta)], axis=-1)


def _fisheye_to_direction(theta, u, v, r):
    with np.errstate(divide="ignore", invalid="ignore"):
        ratio = np.where(r != 0.0, u / np.where(r != 0.0, r, 1.0), 0.0)
    phi = np.arccos(np.clip(ratio, -1.0, 1.0))
    phi = np.where(v < 0.0, -phi, phi)
    return np.stack([np.cos(theta), -np.cos(phi) * np.sin(theta), np.sin(phi) * np.sin(theta)], axis=-1)


def _sensor_size(s):
    """Cycles' sensor width and height for the equisolid and polynomial fisheyes."""
    fit, sensor_width, sensor_height = s["sensor"]
    width, height = s["size"]
    xratio = width * s["pixel_aspect"][0]
    yratio = height * s["pixel_aspect"][1]
    if fit == "AUTO":
        horizontal, size = xratio > yratio, sensor_width
    elif fit == "HORIZONTAL":
        horizontal, size = True, sensor_width
    else:
        horizontal, size = False, sensor_height
    if horizontal:
        return size, size * yratio / xratio
    return size * xratio / yratio, size


def _panorama_to_direction(s, u, v):
    kind = s["type"]
    zero = np.zeros(u.shape + (3,))
    if kind == "EQUIRECTANGULAR":
        lat_min, lat_max = s["latitude"]
        lon_min, lon_max = s["longitude"]
        phi = (lon_min - lon_max) * u - lon_min
        theta = (lat_min - lat_max) * v - lat_min + np.pi / 2
        return _spherical_to_direction(theta, phi)
    if kind == "EQUIANGULAR_CUBEMAP_FACE":
        d = np.stack([np.ones_like(u), np.tan((0.5 - u) * np.pi / 2), np.tan((v - 0.5) * np.pi / 2)], axis=-1)
        return d / np.linalg.norm(d, axis=-1, keepdims=True)
    if kind == "MIRRORBALL":
        x, z = 2.0 * u - 1.0, 2.0 * v - 1.0
        inside = x * x + z * z <= 1.0
        y = -np.sqrt(np.maximum(1.0 - x * x - z * z, 0.0))
        # the reflection of I = (0, -1, 0) off the ball's surface point
        dot = -y
        d = np.stack([2.0 * dot * x, 2.0 * dot * y + 1.0, 2.0 * dot * z], axis=-1)
        return np.where(inside[..., None], d, zero)
    if kind == "FISHEYE_EQUIDISTANT":
        uu, vv = (u - 0.5) * 2.0, (v - 0.5) * 2.0
        r = np.sqrt(uu * uu + vv * vv)
        d = _fisheye_to_direction(r * s["fisheye_fov"] * 0.5, uu, vv, r)
        return np.where((r <= 1.0)[..., None], d, zero)
    if kind == "FISHEYE_LENS_POLYNOMIAL":
        width, height = _sensor_size(s)
        uu, vv = (u - 0.5) * width, (v - 0.5) * height
        r = np.sqrt(uu * uu + vv * vv)
        k0, k1, k2, k3, k4 = s["polynomial"]
        theta = -(k0 + k1 * r + k2 * r**2 + k3 * r**3 + k4 * r**4)
        d = _fisheye_to_direction(theta, uu, vv, r)
        return np.where((np.abs(theta) <= 0.5 * s["fisheye_fov"])[..., None], d, zero)
    if kind == "CENTRAL_CYLINDRICAL":
        u_min, u_max, v_min, v_max, radius = s["cylindrical"]
        theta = -u_min + (u_min - u_max) * u
        z = v_min / radius + (v_max - v_min) / radius * v
        return np.stack([np.cos(theta), np.sin(theta), z], axis=-1)
    # FISHEYE_EQUISOLID, also Cycles' default
    width, height = _sensor_size(s)
    lens, fov = s["fisheye_lens"], s["fisheye_fov"]
    uu, vv = (u - 0.5) * width, (v - 0.5) * height
    r = np.sqrt(uu * uu + vv * vv)
    inside = r <= 2.0 * lens * np.sin(fov * 0.25)
    theta = 2.0 * np.arcsin(np.clip(r / (2.0 * lens), -1.0, 1.0))
    d = _fisheye_to_direction(theta, uu, vv, r)
    return np.where(inside[..., None], d, zero)


def _spherical_stereo(s, d, offset):
    """(origins, directions) of one eye; Cycles' spherical_stereo_transform."""
    d = d / np.maximum(np.linalg.norm(d, axis=-1, keepdims=True), 1e-30)
    offsets = np.full(d.shape[:-1], offset)
    if s["pole_merge"] is not None:
        merge_from, merge_to = s["pole_merge"]
        altitude = np.abs(np.arcsin(np.clip(d[..., 2], -1.0, 1.0)))
        fade = np.cos((altitude - merge_from) / (merge_to - merge_from) * np.pi / 2)
        offsets = np.where(altitude > merge_to, 0.0, np.where(altitude > merge_from, offsets * fade, offsets))
    side = np.cross(d, (0.0, 0.0, 1.0))
    side /= np.maximum(np.linalg.norm(side, axis=-1, keepdims=True), 1e-30)
    p = side * offsets[..., None]
    if s["convergence"] is not None:
        d = s["convergence"] * d - p
    return p, d
