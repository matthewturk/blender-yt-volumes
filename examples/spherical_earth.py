"""A synthetic whole-mantle dataset in yt's spherical geometry, for testing.

Plume columns rise from the core-mantle boundary under three hotspots, so
their positions can be checked against geography drawn in Blender. Radii are
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


def load(shape=(48, 256, 128), nprocs=64):
    """The dataset, with field ("stream", "plumes") in [0, 1]; shape is (r, phi, theta)."""
    import yt

    bbox = np.array([[CMB_RADIUS, 1.0], [0.0, 2 * np.pi], [0.0, np.pi]])
    edges = [np.linspace(lo, hi, n + 1) for (lo, hi), n in zip(bbox, shape)]
    r, phi, theta = np.meshgrid(*[(e[1:] + e[:-1]) / 2 for e in edges], indexing="ij")
    points = np.stack([np.sin(theta) * np.cos(phi), np.sin(theta) * np.sin(phi), np.cos(theta)], axis=-1)
    plumes = np.zeros(shape)
    for lat, lon in HOTSPOTS.values():
        cos_angle = np.clip(points @ unit_vector(lat, lon), -1.0, 1.0)
        angle = np.degrees(np.arccos(cos_angle))
        # columns that widen into a head near the surface
        width = PLUME_WIDTH_DEG * (1.0 + 2.0 * ((r - CMB_RADIUS) / (1.0 - CMB_RADIUS)) ** 4)
        plumes = np.maximum(plumes, np.exp(-((angle / width) ** 2)))
    # a weak, radially decreasing background so the shell itself shows
    background = 0.05 * (1.0 - (r - CMB_RADIUS) / (1.0 - CMB_RADIUS))
    data = {"plumes": plumes + background}
    return yt.load_uniform_grid(
        data, shape, bbox=bbox, nprocs=nprocs, geometry="spherical",
        axis_order=("r", "phi", "theta"), length_unit=1,
    )
