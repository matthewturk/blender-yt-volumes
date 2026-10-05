"""Drives the real yt_idv renderers with a simulated Cycles Z pass (a solid
disc occluder in front of the volume) to check the depth clip end to end.

Run inside the throwaway venv (see setup_ytidv_venv.sh). Output
(images/22_*.png, images/23_*.png) shows a clean disc-shaped hole in the volume.
"""

import math

import numpy as np
import yt

ADDON_SRC = "/home/mturk/yt/cycles-yt/src/yt_volumes"


def _load_addon_modules():
    import importlib.util
    import sys
    import types

    pkg = types.ModuleType("yt_volumes")
    pkg.__path__ = [ADDON_SRC]
    sys.modules.setdefault("yt_volumes", pkg)

    def load(name):
        spec = importlib.util.spec_from_file_location(f"yt_volumes.{name}", f"{ADDON_SRC}/{name}.py")
        mod = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = mod
        spec.loader.exec_module(mod)
        return mod

    return load("external_render"), load("yt_idv_renderer")


def _perspective_matrix(fovy_deg, aspect, near, far):
    """OpenGL-style perspective matrix, standing in for `Object.calc_matrix_camera()`."""
    f = 1.0 / math.tan(math.radians(fovy_deg) / 2.0)
    return [
        [f / aspect, 0, 0, 0],
        [0, f, 0, 0],
        [0, 0, (far + near) / (near - far), (2 * far * near) / (near - far)],
        [0, 0, -1, 0],
    ]


def main():
    er, yir = _load_addon_modules()

    width, height = 256, 192

    camera = er.CameraParams(
        matrix_world=[[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 3], [0, 0, 0, 1]],
        projection="PERSP", lens_mm=50.0, sensor_width_mm=36.0, sensor_height_mm=27.0,
        sensor_fit="AUTO", shift_x=0.0, shift_y=0.0, ortho_scale=None,
        clip_start=0.1, clip_end=100.0,
        resolution_x=width, resolution_y=height, pixel_aspect_x=1.0, pixel_aspect_y=1.0,
        projection_matrix=_perspective_matrix(39.6, width / height, 0.1, 100.0),
        panorama=None,
        dof=er.DepthOfFieldParams(
            use_dof=False, focus_distance_m=5.0, aperture_fstop=2.8,
            aperture_blades=0, aperture_ratio=1.0, aperture_rotation_rad=0.0,
        ),
    )

    # Simulated Z pass: disc at depth 1.5, in front of the volume (z in [-1, 1]).
    no_hit = 1e10
    depth_data = np.full((height, width), no_hit, dtype=np.float32)
    cy, cx = height // 2, width // 2
    yy, xx = np.mgrid[0:height, 0:width]
    disc = ((yy - cy) / (height * 0.25)) ** 2 + ((xx - cx) / (height * 0.25)) ** 2 <= 1.0
    depth_data[disc] = 1.5
    depth = er.DepthBuffer(
        data=depth_data, depth_convention="PLANAR", no_hit_value=no_hit,
        row_order="TOP_LEFT_ORIGIN", units="meters",
    )

    version = er.VersionInfo(hash="x", changed=True)
    dataset = er.DatasetParams(
        dataset_id="obj", dataset_kind="yt", dataset_path=None,
        is_blender_volume_sequence=False, sequence_frame_start=None,
        sequence_frame_duration=None, sequence_frame_offset=None,
        sequence_mode=None, resolved_sequence_frame=None, grid_names=[],
        version=version,
    )
    volume = er.VolumeObjectParams(
        matrix_world=[[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]],
        bounding_box_local=[[x, y, z] for x in (-1, 1) for y in (-1, 1) for z in (-1, 1)],
        custom_params={}, params_version=version, dataset=dataset,
    )
    frame = er.FrameParams(frame_number=1, scene_time_seconds=0.04)
    request = er.RenderRequest(camera=camera, depth=depth, volume=volume, frame=frame)

    result = yir.yt_idv_max_intensity_renderer(request)
    print("MIP result shape:", result.rgba.shape)
    yt.write_bitmap(result.rgba[:, :, :3], "22_cycles_zbuffer_disc_occlusion.png")

    # Projection gives a smoother fill than MIP for this scene.
    result_proj = yir.yt_idv_projection_renderer(request)
    print("projection result shape:", result_proj.rgba.shape)
    yt.write_bitmap(result_proj.rgba[:, :, :3], "23_cycles_zbuffer_disc_occlusion_projection.png")


if __name__ == "__main__":
    main()
