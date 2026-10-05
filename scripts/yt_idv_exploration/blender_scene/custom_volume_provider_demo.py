"""Checks the `yt_idv_volume_provider` hook in `yt_idv_renderer.py` against
real yt/yt_idv (no Cycles render; `bpy` is only used to build a real camera
projection matrix). Run with:

    env -u LD_PRELOAD uv run --group ytidv python \\
        scripts/yt_idv_exploration/blender_scene/custom_volume_provider_demo.py

Claims checked:

  1. A custom provider loading the same `IsolatedGalaxy` sample renders a
     pixel-identical image to the default provider.
  2. A second, unchanged call invokes the provider again with
     `ctx.previous_scene` set.
  3. An unchanged call returns the exact cached `RenderResult` (render
     skipped); a camera move forces a real re-render.
  4. With two objects sharing one render_context, `rc.scene` is re-pointed
     at each object's own scene on every render.
"""

import os
import sys

import bpy
import mathutils
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.normpath(os.path.join(HERE, "..", "..", ".."))
ADDON_SRC_PARENT = os.path.join(REPO_ROOT, "src")
sys.path.insert(0, ADDON_SRC_PARENT)

from cycles_yt import operators  # noqa: E402
from cycles_yt import yt_idv_renderer as yir  # noqa: E402
from cycles_yt.external_render import (  # noqa: E402
    DatasetParams,
    DepthBuffer,
    FrameParams,
    RenderRequest,
    VersionInfo,
    VolumeObjectParams,
)

RENDER_SIZE = 64  # small -- this is checking plumbing/caching, not image quality

bpy.ops.wm.read_factory_settings(use_empty=True)
bpy.context.scene.render.resolution_x = RENDER_SIZE
bpy.context.scene.render.resolution_y = RENDER_SIZE
bpy.ops.object.camera_add()
_CAMERA_OBJ = bpy.context.active_object


def _camera(matrix_world):
    """Real `CameraParams`, including the projection matrix the depth clip needs."""
    # bpy silently ignores a plain nested list for matrix_world; needs a Matrix.
    _CAMERA_OBJ.matrix_world = mathutils.Matrix(matrix_world)
    bpy.context.view_layer.update()
    return operators.build_camera_params(bpy.context.scene, _CAMERA_OBJ)


def _request(dataset_id, matrix_world, custom_params):
    version = VersionInfo(hash="x", changed=True)
    volume = VolumeObjectParams(
        matrix_world=[[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]],
        bounding_box_local=[
            [x, y, z] for x in (-1, 1) for y in (-1, 1) for z in (-1, 1)
        ],
        custom_params=custom_params,
        params_version=version,
        dataset=DatasetParams(
            dataset_id=dataset_id, dataset_kind="", dataset_path=None,
            is_blender_volume_sequence=False, sequence_frame_start=None,
            sequence_frame_duration=None, sequence_frame_offset=None,
            sequence_mode=None, resolved_sequence_frame=None, grid_names=[],
            version=version,
        ),
    )
    return RenderRequest(
        camera=_camera(matrix_world),
        depth=DepthBuffer(
            data=np.zeros((RENDER_SIZE, RENDER_SIZE), dtype=np.float32),
            depth_convention="EUCLIDEAN", no_hit_value=1e10,
            row_order="TOP_LEFT_ORIGIN", units="meters",
        ),
        volume=volume,
        frame=FrameParams(frame_number=1, scene_time_seconds=0.0),
    )


CAMERA_AT_ORIGIN_OFFSET = [[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 4], [0, 0, 0, 1]]
CAMERA_MOVED = [[1, 0, 0, 1.5], [0, 1, 0, 0], [0, 0, 1, 4], [0, 0, 0, 1]]

sys.path.insert(0, HERE)  # so "_custom_volume_provider_example:setup_volume" resolves
import _custom_volume_provider_example as provider_module  # noqa: E402


def main():
    custom_params_default = {}
    custom_params_custom = {
        "yt_idv_volume_provider": "_custom_volume_provider_example:setup_volume",
    }

    # --- Claim 1: identical output to the default provider -----------------
    default_request = _request("default-obj", CAMERA_AT_ORIGIN_OFFSET, custom_params_default)
    default_result = yir.yt_idv_projection_renderer(default_request)

    custom_request = _request("custom-obj", CAMERA_AT_ORIGIN_OFFSET, custom_params_custom)
    custom_result = yir.yt_idv_projection_renderer(custom_request)

    assert np.array_equal(default_result.rgba, custom_result.rgba), (
        "custom provider's output differs from the default provider's -- "
        "the harness itself is changing what gets rendered"
    )
    print("[custom_volume_provider_demo] claim 1 OK: custom provider's image "
          "is pixel-identical to the default provider's")

    # --- Claim 2: provider called again, told to reuse its own scene -------
    assert provider_module.calls == [False]  # first call: previous_scene was None
    repeat_result = yir.yt_idv_projection_renderer(custom_request)
    assert provider_module.calls == [False, True], (
        f"expected the provider's second call to see previous_scene set, got {provider_module.calls}"
    )
    print("[custom_volume_provider_demo] claim 2 OK: second call passed "
          "previous_scene through (reused, not silently rebuilt)")

    # --- Claim 3: skip-cache returns the identical object; a real change ---
    #              (camera move) does NOT reuse it, even with changed=False
    assert repeat_result is custom_result, (
        "unchanged second call did not return the exact cached RenderResult "
        "-- the skip-render cache did not actually skip rc.run()"
    )
    print("[custom_volume_provider_demo] claim 3a OK: unchanged call returned "
          "the exact same RenderResult object (rc.run() genuinely skipped)")

    moved_request = _request("custom-obj", CAMERA_MOVED, custom_params_custom)
    moved_result = yir.yt_idv_projection_renderer(moved_request)
    assert moved_result is not custom_result, (
        "camera-moved request returned the SAME cached RenderResult -- "
        "the skip-render cache is over-caching across a real visual change"
    )
    assert not np.array_equal(moved_result.rgba, custom_result.rgba), (
        "camera moved but the rendered image is pixel-identical -- either "
        "the camera move had no visual effect, or a re-render didn't "
        "actually happen despite passing the object-identity check above"
    )
    print("[custom_volume_provider_demo] claim 3b OK: camera-only change "
          "still forced a real, different re-render")

    # --- Claim 4: rc.scene re-pointed at the default object's scene -------
    #              (camera move so the skip-render cache doesn't short-circuit)
    default_moved_request = _request("default-obj", CAMERA_MOVED, custom_params_default)
    yir.yt_idv_projection_renderer(default_moved_request)
    rc_now, cached_default_scene, _ds, _changed = yir._get_volume_scene(default_moved_request, RENDER_SIZE)
    assert rc_now.scene is cached_default_scene, (
        "after re-rendering the default object, rc.scene points at a "
        "DIFFERENT scene than the default object's own cached one -- it's "
        "still left over from the custom object's render in between"
    )
    print("[custom_volume_provider_demo] claim 4 OK: default object's scene "
          "correctly re-selected on a shared render_context")

    print("[custom_volume_provider_demo] all claims verified")


if __name__ == "__main__":
    main()
