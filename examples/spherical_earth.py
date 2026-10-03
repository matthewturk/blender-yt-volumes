"""A synthetic whole-mantle dataset in yt's spherical geometry, for testing.

Plumes rise from the core-mantle boundary under three hotspots, so their
positions can be checked against geography drawn in Blender. Radii are
in units of Earth's radius; theta is co-latitude from +z (north), phi is east
longitude from +x.
"""

import numpy as np

CMB_RADIUS = 3480.0 / 6371.0
# name: (latitude, east longitude), degrees
HOTSPOTS = {"Hawaii": (19.6, -155.5), "Iceland": (64.4, -17.3), "Reunion": (-21.1, 55.5)}
PLUME_WIDTH_DEG = 6.0


def unit_vector(lat_deg, lon_deg):
    lat, lon = np.radians(lat_deg), np.radians(lon_deg)
    return np.array([np.cos(lat) * np.cos(lon), np.cos(lat) * np.sin(lon), np.sin(lat)])


def plumes_at(t, r, points):
    """The plume field at time `t` in [0, 1]: heads rise from the core-mantle
    boundary one after another, trailing narrow tails, and spread out under
    the surface once they arrive.
    """
    field = np.zeros(r.shape)
    depth = (r - CMB_RADIUS) / (1.0 - CMB_RADIUS)  # 0 at the CMB, 1 at the surface
    for i, (lat, lon) in enumerate(HOTSPOTS.values()):
        start, rise = 0.12 * i, 0.5
        s = np.clip((t - start) / rise, 0.0, 1.0)
        s = s * s * (3.0 - 2.0 * s)  # ease in and out
        arrived = np.clip((t - start - rise) / 0.3, 0.0, 1.0)
        head_depth = 0.08 + 0.84 * s
        angle = np.degrees(np.arccos(np.clip(points @ unit_vector(lat, lon), -1.0, 1.0)))
        tail = np.exp(-((angle / (0.7 * PLUME_WIDTH_DEG)) ** 2)) / (1.0 + np.exp((depth - head_depth) / 0.02))
        head_width = PLUME_WIDTH_DEG * (1.6 + 1.2 * arrived)
        head_thickness = 0.12 * (1.0 - 0.5 * arrived)
        head = np.exp(-(((depth - head_depth) / head_thickness) ** 2) - (angle / head_width) ** 2)
        field = np.maximum(field, np.maximum(0.8 * tail, head))
    return field


def load(shape=(48, 256, 128), nprocs=64, frame=None, frames=250):
    """The dataset, with field ("stream", "plumes") in [0, 1]; shape is (r, phi, theta).

    `frame` (1 to `frames`) picks a moment of the plumes' rise; without it,
    a late one, after all three heads have reached the surface.
    """
    import yt

    t = 0.9 if frame is None else (frame - 1) / max(frames - 1, 1)
    bbox = np.array([[CMB_RADIUS, 1.0], [0.0, 2 * np.pi], [0.0, np.pi]])
    edges = [np.linspace(lo, hi, n + 1) for (lo, hi), n in zip(bbox, shape)]
    r, phi, theta = np.meshgrid(*[(e[1:] + e[:-1]) / 2 for e in edges], indexing="ij")
    points = np.stack([np.sin(theta) * np.cos(phi), np.sin(theta) * np.sin(phi), np.cos(theta)], axis=-1)
    # a weak, radially decreasing background so the shell itself shows
    background = 0.05 * (1.0 - (r - CMB_RADIUS) / (1.0 - CMB_RADIUS))
    data = {"plumes": plumes_at(t, r, points) + background}
    return yt.load_uniform_grid(
        data, shape, bbox=bbox, nprocs=nprocs, geometry="spherical",
        axis_order=("r", "phi", "theta"), length_unit=1,
    )
