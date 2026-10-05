"""Exercises `yt_idv_renderer.yt_idv_max_intensity_renderer` end to end with a
hand-built `RenderRequest`, including non-square resolutions and a camera
change between calls.

Run inside the throwaway venv (see setup_ytidv_venv.sh). Add-on modules are
loaded via `importlib` because the package `__init__.py` imports bpy.
"""

import importlib.util
import math
import sys
import types


def _load_addon_module(name, path):
    """Load an add-on module without running the package's bpy-dependent `__init__.py`."""
    spec = importlib.util.spec_from_file_location(f"yt_volumes.{name}", path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod


def _matrix_world_pitched(pos, pitch_x_rad):
    """Camera matrix_world at `pos`, pitched by `pitch_x_rad` about local X."""
    c, s = math.cos(pitch_x_rad), math.sin(pitch_x_rad)
    rot = [[1, 0, 0], [0, c, -s], [0, s, c]]
    return [
        [rot[0][0], rot[0][1], rot[0][2], pos[0]],
        [rot[1][0], rot[1][1], rot[1][2], pos[1]],
        [rot[2][0], rot[2][1], rot[2][2], pos[2]],
        [0, 0, 0, 1],
    ]


def main():
    pkg = types.ModuleType("yt_volumes")
    pkg.__path__ = [ADDON_SRC]
    sys.modules["yt_volumes"] = pkg

    er = _load_addon_module("external_render", f"{ADDON_SRC}/external_render.py")
    yir = _load_addon_module("yt_idv_renderer", f"{ADDON_SRC}/yt_idv_renderer.py")

    import numpy as np
    import yt

    width, height = 200, 150

    camera = er.CameraParams(
        matrix_world=_matrix_world_pitched((0.0, -5.0, 0.0), math.radians(90)),
        projection="PERSP", lens_mm=50.0,
        sensor_width_mm=36.0, sensor_height_mm=24.0, sensor_fit="AUTO",
        shift_x=0.0, shift_y=0.0, ortho_scale=None,
        clip_start=0.1, clip_end=1000.0,
        resolution_x=width, resolution_y=height,
        pixel_aspect_x=1.0, pixel_aspect_y=1.0,
        projection_matrix=None, panorama=None,
        dof=er.DepthOfFieldParams(
            use_dof=False, focus_distance_m=5.0, aperture_fstop=2.8,
            aperture_blades=0, aperture_ratio=1.0, aperture_rotation_rad=0.0,
        ),
    )
    depth = er.DepthBuffer(
        data=np.full((height, width), 1e10, dtype=np.float32),
        depth_convention="PLANAR", no_hit_value=1e10,
        row_order="BOTTOM_LEFT_ORIGIN", units="meters",
    )
    version = er.VersionInfo(hash="x", changed=True)
    dataset = er.DatasetParams(
        dataset_id="obj1", dataset_kind="yt", dataset_path=None,
        is_blender_volume_sequence=False, sequence_frame_start=None,
        sequence_frame_duration=None, sequence_frame_offset=None,
        sequence_mode=None, resolved_sequence_frame=None, grid_names=[],
        version=version,
    )
    volume = er.VolumeObjectParams(
        matrix_world=[[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]],
        bounding_box_local=[[0, 0, 0]] * 8, custom_params={},
        params_version=version, dataset=dataset,
    )
    frame = er.FrameParams(frame_number=1, scene_time_seconds=0.0417)
    request = er.RenderRequest(camera=camera, depth=depth, volume=volume, frame=frame)

    result = yir.yt_idv_max_intensity_renderer(request)
    print("result.rgba.shape (expect exactly (H, W, 4) =", (height, width, 4), "):", result.rgba.shape)
    print("alpha_mode:", result.alpha_mode, "color_space:", result.color_space)
    yt.write_bitmap(result.rgba[:, :, :3], "03_stub_end_to_end_200x150_persp90deg.png")

    import dataclasses
    camera2 = dataclasses.replace(
        camera, matrix_world=_matrix_world_pitched((0.0, -5.0, 0.0), math.radians(60)),
    )
    request2 = er.RenderRequest(camera=camera2, depth=depth, volume=volume, frame=frame)
    result2 = yir.yt_idv_max_intensity_renderer(request2)
    print("differs after changing Blender camera pitch:", not np.allclose(result.rgba, result2.rgba))
    yt.write_bitmap(result2.rgba[:, :, :3], "04_stub_end_to_end_200x150_persp60deg.png")

    # Extreme aspect ratio exercises _fix_yt_idv_axis_order.
    wide_camera = dataclasses.replace(
        camera,
        resolution_x=400, resolution_y=100,
        matrix_world=_matrix_world_pitched((0.0, -5.0, 0.0), math.radians(75)),
    )
    wide_depth = dataclasses.replace(depth, data=np.full((100, 400), 1e10, dtype=np.float32))
    wide_request = er.RenderRequest(camera=wide_camera, depth=wide_depth, volume=volume, frame=frame)
    wide_result = yir.yt_idv_max_intensity_renderer(wide_request)
    print("wide result shape (expect (100, 400, 4)):", wide_result.rgba.shape)
    yt.write_bitmap(wide_result.rgba[:, :, :3], "09_extreme_aspect_via_real_module_fixed.png")


if __name__ == "__main__":
    import os

    ADDON_SRC = os.path.normpath(
        os.path.join(os.path.dirname(__file__), "..", "..", "src", "yt_volumes")
    )
    main()
