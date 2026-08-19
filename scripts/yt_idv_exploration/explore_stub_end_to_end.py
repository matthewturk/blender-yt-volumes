"""Third exploration: exercise the REAL add-on module
(`cycles_volume_override.yt_idv_renderer.yt_idv_max_intensity_renderer`)
end to end with a hand-built `RenderRequest` standing in for what
`operators.py` would actually construct from a Blender scene -- not just
calling yt_idv's own API directly like the first two scripts did.

Run inside the throwaway venv (see setup_ytidv_venv.sh). Loads
`external_render.py`/`yt_idv_renderer.py` directly via `importlib`, NOT
`import cycles_volume_override` -- that package's `__init__.py` eagerly
imports every submodule including bpy-dependent ones
(compositor_build.py, properties.py, ...), which aren't installed/importable
here (no real Blender, no `bpy` in this venv -- see yt_idv_renderer.py's
module docstring for why `bpy`/`yt_idv` can't coexist in one venv here).

Confirms:
  - The full function signature works: build a `RenderRequest`, get back a
    `RenderResult` with `rgba.shape == (H, W, 4)` matching the requested
    depth buffer's own shape exactly (not yt_idv's raw, axis-order-buggy
    shape -- `_fix_yt_idv_axis_order` inside the real function already
    corrected it).
  - Changing the Blender camera's orientation between two calls (same
    dataset, same context, freshly cached) produces genuinely different
    output (`np.allclose` check) -- the orientation-only camera mapping
    (see yt_idv_renderer.py's module docstring) is at least DOING
    something, not silently ignoring the input.
  - Non-square resolutions (200x150, matching a real Cycles-style aspect
    ratio rather than yt_idv's own square examples) round-trip through the
    real module with the correct final shape -- see
    images/03_*.png/04_*.png.
"""

import importlib.util
import math
import sys
import types


def _load_addon_module(name, path):
    """See module docstring: bypasses cycles_volume_override/__init__.py's
    eager bpy-dependent imports.
    """
    spec = importlib.util.spec_from_file_location(f"cycles_volume_override.{name}", path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod


def _matrix_world_pitched(pos, pitch_x_rad):
    """A Blender camera.matrix_world for a camera at `pos`, pitched by
    `pitch_x_rad` about local X (matches how this repo's own tests build
    camera transforms, e.g. tests/test_compositor_build.py).
    """
    c, s = math.cos(pitch_x_rad), math.sin(pitch_x_rad)
    rot = [[1, 0, 0], [0, c, -s], [0, s, c]]
    return [
        [rot[0][0], rot[0][1], rot[0][2], pos[0]],
        [rot[1][0], rot[1][1], rot[1][2], pos[1]],
        [rot[2][0], rot[2][1], rot[2][2], pos[2]],
        [0, 0, 0, 1],
    ]


def main():
    pkg = types.ModuleType("cycles_volume_override")
    pkg.__path__ = [ADDON_SRC]
    sys.modules["cycles_volume_override"] = pkg

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
        requested_time=0.0, time_units="FRAME_INDEX", mapping_policy="DIRECT",
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

    # Extreme aspect ratio, via the real module -- confirms
    # _fix_yt_idv_axis_order actually gets exercised correctly end-to-end,
    # not just in explore_axis_order_bug.py's more manual reproduction.
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
        os.path.join(os.path.dirname(__file__), "..", "..", "src", "cycles_volume_override")
    )
    main()
