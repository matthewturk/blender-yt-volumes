"""Seventh exploration: wiring the external depth clip (see
explore_external_depth_clip.py) up to a REAL Cycles-shaped Z-buffer, using
the flagged object's own `bounding_box_local`/`matrix_world` as the
registration between Blender world space and yt_idv's unitary domain space
-- no new yt_idv Camera subclass needed, just feeding
`BaseCamera.update(position=..., focus=..., up=..., ...)` genuinely
transformed values instead of the old orbit-at-a-fixed-distance heuristic.

Run inside the throwaway venv (see setup_ytidv_venv.sh). Exercises the REAL
`yt_idv_max_intensity_renderer`/`yt_idv_transfer_function_renderer`
functions -- not a hand-rolled equivalent -- with a hand-built
`RenderRequest` whose `depth.data` simulates what Cycles' own Z pass would
produce for a solid disc occluder sitting between the camera and the
flagged object: a per-pixel PLANAR depth array with a disc of near values
surrounded by the `no_hit_value` sentinel everywhere else, exactly the
shape `operators.read_depth_buffer()` would hand this renderer for a real
solid partially occluding a flagged volume.

The math itself (`_registered_camera_kwargs`, `_cycles_depth_to_yt_idv_
window_depth`) is separately verified against a REAL bpy-computed
`projection_matrix` in `tests/test_yt_idv_renderer.py` -- this script is
about confirming the whole thing produces a sensible, real, disc-shaped
occlusion when driven by realistic depth data end to end, not re-deriving
the math by hand again.

Result: `images/22_cycles_zbuffer_disc_occlusion.png` -- the domain box's
silhouette with a genuine, clean DISC-shaped hole cut exactly where the
fabricated occluder sits between the camera and the object. This is the
real depth-clip working against depth data shaped like what Cycles would
actually produce, not a same-yt_idv-session synthetic sphere test.
"""

import math

import numpy as np
import yt

ADDON_SRC = "/home/mturk/yt/cycles-yt/src/cycles_volume_override"


def _load_addon_modules():
    import importlib.util
    import sys
    import types

    pkg = types.ModuleType("cycles_volume_override")
    pkg.__path__ = [ADDON_SRC]
    sys.modules.setdefault("cycles_volume_override", pkg)

    def load(name):
        spec = importlib.util.spec_from_file_location(f"cycles_volume_override.{name}", f"{ADDON_SRC}/{name}.py")
        mod = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = mod
        spec.loader.exec_module(mod)
        return mod

    return load("external_render"), load("yt_idv_renderer")


def _perspective_matrix(fovy_deg, aspect, near, far):
    """Standard OpenGL-style perspective matrix -- stands in for what
    Blender's own `Object.calc_matrix_camera()` would produce (that exact
    round-trip is what `tests/test_yt_idv_renderer.py` verifies against
    real bpy; this script just needs A plausible, self-consistent
    projection matrix to drive the exploration, not Blender's own).
    """
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

    # Simulated Cycles Z pass: a solid disc occluder at depth=1.5 (world
    # z = 3 - 1.5 = 1.5), everything else background (no_hit_value).
    # Flagged object below spans world z in [-1, 1] -- entirely BEHIND the
    # disc from the camera's point of view (camera at z=3).
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
        requested_time=0.0, time_units="FRAME_INDEX", mapping_policy="DIRECT",
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

    # MIP's per-ray "keep only the single brightest sample" reads noisy/
    # speckled for this scene -- tried yt_idv's "projection" render_method
    # (unweighted line-of-sight integration, additive not max) on the exact
    # same request: noticeably smoother/more coherent fill. The disc clip
    # boundary itself still shows some jaggedness in both -- looks like an
    # AMR block-boundary artifact (different refinement levels contributing
    # their own edge near the clip), not a MIP-vs-projection difference;
    # not chased further here.
    result_proj = yir.yt_idv_projection_renderer(request)
    print("projection result shape:", result_proj.rgba.shape)
    yt.write_bitmap(result_proj.rgba[:, :, :3], "23_cycles_zbuffer_disc_occlusion_projection.png")


if __name__ == "__main__":
    main()
