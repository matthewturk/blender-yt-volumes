"""HISTORICAL: subprocess worker bridging this repo's bpy/Python-3.13 venv
to a separate yt_idv/Python-3.12 venv (see ../setup_ytidv_venv.sh). No
longer needed: since yt_idv#268 made imgui optional, bpy and yt_idv share
one venv (`uv sync --group ytidv`). Not part of the shipped add-on.

Run inside the yt_idv venv:

    describe_grids: dump the sample dataset's AMR grid boxes (unitary
    [0, 1]^3) as JSON.

        python ytidv_worker.py describe_grids --out grids.json --max-grids 40

    render: render one JSON RenderRequest with
    yt_idv_max_intensity_renderer() to a PNG.

        python ytidv_worker.py render --request request.json --out frame.png

    render_many: render a JSON list of requests in one process (used by
    render_frames_in_blender.py).

    render_with_depth: like `render`, but with a real per-pixel depth array
    (.npy float32) and a choice of --mode; writes float RGBA .npy so
    premultiplied alpha survives.

        python ytidv_worker.py render_with_depth --request request.json \\
            --depth-npy depth.npy --mode projection --out-npy result.npy
"""

import argparse
import importlib.util
import json
import os
import sys
import types

HERE = os.path.dirname(os.path.abspath(__file__))
ADDON_SRC = os.path.normpath(os.path.join(HERE, "..", "..", "..", "src", "yt_volumes"))


def _load_addon_module(name, path):
    """Load one add-on submodule, skipping the package's bpy-dependent __init__."""
    spec = importlib.util.spec_from_file_location(f"yt_volumes.{name}", path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod


def _addon_modules():
    pkg = types.ModuleType("yt_volumes")
    pkg.__path__ = [ADDON_SRC]
    sys.modules.setdefault("yt_volumes", pkg)
    external_render = _load_addon_module("external_render", os.path.join(ADDON_SRC, "external_render.py"))
    yt_idv_renderer = _load_addon_module("yt_idv_renderer", os.path.join(ADDON_SRC, "yt_idv_renderer.py"))
    return external_render, yt_idv_renderer


def cmd_describe_grids(args):
    import yt

    external_render, yt_idv_renderer = _addon_modules()
    ds, _ = yt_idv_renderer._load_sample_source()

    grids = []
    for g in ds.index.grids[: args.max_grids]:
        grids.append({
            "level": int(g.Level),
            "left_edge": yt_idv_renderer._domain_array_unitary(g.LeftEdge).tolist(),
            "right_edge": yt_idv_renderer._domain_array_unitary(g.RightEdge).tolist(),
        })
    domain = {
        "center": yt_idv_renderer._domain_array_unitary(ds.domain_center).tolist(),
        "width": yt_idv_renderer._domain_array_unitary(ds.domain_width).tolist(),
    }
    with open(args.out, "w") as f:
        json.dump({"domain": domain, "grids": grids}, f, indent=2)
    print(f"wrote {len(grids)} grids (of {len(ds.index.grids)} total) + domain bbox to {args.out}")


def _build_request(external_render, req):
    import numpy as np

    camera = external_render.CameraParams(
        matrix_world=req["camera"]["matrix_world"],
        projection=req["camera"]["projection"],
        lens_mm=req["camera"]["lens_mm"],
        sensor_width_mm=req["camera"]["sensor_width_mm"],
        sensor_height_mm=req["camera"]["sensor_height_mm"],
        sensor_fit=req["camera"]["sensor_fit"],
        shift_x=req["camera"]["shift_x"],
        shift_y=req["camera"]["shift_y"],
        ortho_scale=req["camera"]["ortho_scale"],
        clip_start=req["camera"]["clip_start"],
        clip_end=req["camera"]["clip_end"],
        resolution_x=req["camera"]["resolution_x"],
        resolution_y=req["camera"]["resolution_y"],
        pixel_aspect_x=req["camera"]["pixel_aspect_x"],
        pixel_aspect_y=req["camera"]["pixel_aspect_y"],
        projection_matrix=None,
        panorama=None,
        dof=external_render.DepthOfFieldParams(
            use_dof=False, focus_distance_m=5.0, aperture_fstop=2.8,
            aperture_blades=0, aperture_ratio=1.0, aperture_rotation_rad=0.0,
        ),
    )
    width, height = req["camera"]["resolution_x"], req["camera"]["resolution_y"]
    depth = external_render.DepthBuffer(
        data=np.full((height, width), 1e10, dtype=np.float32),
        depth_convention="PLANAR", no_hit_value=1e10,
        row_order=req["row_order"], units="meters",
    )
    version = external_render.VersionInfo(hash="x", changed=True)
    dataset = external_render.DatasetParams(
        dataset_id="obj1", dataset_kind="yt", dataset_path=None,
        is_blender_volume_sequence=False, sequence_frame_start=None,
        sequence_frame_duration=None, sequence_frame_offset=None,
        sequence_mode=None, resolved_sequence_frame=None, grid_names=[],
        version=version,
    )
    volume = external_render.VolumeObjectParams(
        matrix_world=[[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]],
        bounding_box_local=[[0, 0, 0]] * 8, custom_params={},
        params_version=version, dataset=dataset,
    )
    frame = external_render.FrameParams(
        frame_number=req["frame_number"], scene_time_seconds=req["frame_number"] / 24.0,
    )
    return external_render.RenderRequest(camera=camera, depth=depth, volume=volume, frame=frame)


def cmd_render_many(args):
    """Render every request in a JSON list in one process, reusing caches."""
    import time

    import yt

    external_render, yt_idv_renderer = _addon_modules()

    with open(args.requests) as f:
        reqs = json.load(f)

    os.makedirs(args.out_dir, exist_ok=True)
    timings = []
    for req in reqs:
        request = _build_request(external_render, req)
        t0 = time.time()
        result = yt_idv_renderer.yt_idv_max_intensity_renderer(request)
        elapsed = time.time() - t0
        timings.append(elapsed)
        out_path = os.path.join(args.out_dir, f"frame_{req['frame_number']:04d}.png")
        yt.write_bitmap(result.rgba[:, :, :3], out_path)
        print(f"frame {req['frame_number']}: wrote {out_path} shape={result.rgba.shape} ({elapsed:.3f}s)")

    print(f"timings: first={timings[0]:.3f}s, rest avg={sum(timings[1:]) / max(1, len(timings) - 1):.3f}s")


_REAL_RENDER_MODES = None


def _build_request_full(external_render, req, depth_array):
    """Like `_build_request`, but with camera/volume/depth from a real scene.

    `req["camera"]["projection_matrix"]` should come from
    `operators.build_camera_params()`.
    """
    camera = external_render.CameraParams(
        matrix_world=req["camera"]["matrix_world"],
        projection=req["camera"]["projection"],
        lens_mm=req["camera"]["lens_mm"],
        sensor_width_mm=req["camera"]["sensor_width_mm"],
        sensor_height_mm=req["camera"]["sensor_height_mm"],
        sensor_fit=req["camera"]["sensor_fit"],
        shift_x=req["camera"]["shift_x"],
        shift_y=req["camera"]["shift_y"],
        ortho_scale=req["camera"]["ortho_scale"],
        clip_start=req["camera"]["clip_start"],
        clip_end=req["camera"]["clip_end"],
        resolution_x=req["camera"]["resolution_x"],
        resolution_y=req["camera"]["resolution_y"],
        pixel_aspect_x=req["camera"]["pixel_aspect_x"],
        pixel_aspect_y=req["camera"]["pixel_aspect_y"],
        projection_matrix=req["camera"]["projection_matrix"],
        panorama=None,
        dof=external_render.DepthOfFieldParams(
            use_dof=False, focus_distance_m=5.0, aperture_fstop=2.8,
            aperture_blades=0, aperture_ratio=1.0, aperture_rotation_rad=0.0,
        ),
    )
    depth = external_render.DepthBuffer(
        data=depth_array,
        depth_convention=req["depth"]["depth_convention"],
        no_hit_value=req["depth"]["no_hit_value"],
        row_order=req["depth"]["row_order"],
        units=req["depth"]["units"],
    )
    version = external_render.VersionInfo(hash="x", changed=True)
    dataset = external_render.DatasetParams(
        dataset_id="obj1", dataset_kind="yt", dataset_path=None,
        is_blender_volume_sequence=False, sequence_frame_start=None,
        sequence_frame_duration=None, sequence_frame_offset=None,
        sequence_mode=None, resolved_sequence_frame=None, grid_names=[],
        version=version,
    )
    volume = external_render.VolumeObjectParams(
        matrix_world=req["volume"]["matrix_world"],
        bounding_box_local=req["volume"]["bounding_box_local"],
        custom_params={}, params_version=version, dataset=dataset,
    )
    frame = external_render.FrameParams(
        frame_number=req["frame_number"], scene_time_seconds=req["frame_number"] / 24.0,
    )
    return external_render.RenderRequest(camera=camera, depth=depth, volume=volume, frame=frame)


def cmd_render_with_depth(args):
    import numpy as np

    external_render, yt_idv_renderer = _addon_modules()
    render_modes = {
        "max_intensity": yt_idv_renderer.yt_idv_max_intensity_renderer,
        "max_intensity_alpha": yt_idv_renderer.yt_idv_max_intensity_alpha_renderer,
        "transfer_function": yt_idv_renderer.yt_idv_transfer_function_renderer,
        "projection": yt_idv_renderer.yt_idv_projection_renderer,
    }
    with open(args.request) as f:
        req = json.load(f)
    depth_array = np.load(args.depth_npy)
    request = _build_request_full(external_render, req, depth_array)
    result = render_modes[args.mode](request)
    np.save(args.out_npy, result.rgba.astype("float32"))
    print(
        f"wrote {args.out_npy} shape={result.rgba.shape} "
        f"alpha_mode={result.alpha_mode} color_space={result.color_space}"
    )


def cmd_render(args):
    import yt

    external_render, yt_idv_renderer = _addon_modules()
    with open(args.request) as f:
        req = json.load(f)
    request = _build_request(external_render, req)
    result = yt_idv_renderer.yt_idv_max_intensity_renderer(request)
    yt.write_bitmap(result.rgba[:, :, :3], args.out)
    print(f"wrote {args.out} shape={result.rgba.shape}")


def main():
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="command", required=True)

    p_grids = sub.add_parser("describe_grids")
    p_grids.add_argument("--out", required=True)
    p_grids.add_argument("--max-grids", type=int, default=40)
    p_grids.set_defaults(func=cmd_describe_grids)

    p_render = sub.add_parser("render")
    p_render.add_argument("--request", required=True)
    p_render.add_argument("--out", required=True)
    p_render.set_defaults(func=cmd_render)

    p_render_many = sub.add_parser("render_many")
    p_render_many.add_argument("--requests", required=True)
    p_render_many.add_argument("--out-dir", required=True)
    p_render_many.set_defaults(func=cmd_render_many)

    p_render_depth = sub.add_parser("render_with_depth")
    p_render_depth.add_argument("--request", required=True)
    p_render_depth.add_argument("--depth-npy", required=True)
    p_render_depth.add_argument("--mode", choices=["max_intensity", "max_intensity_alpha", "transfer_function", "projection"], default="projection")
    p_render_depth.add_argument("--out-npy", required=True)
    p_render_depth.set_defaults(func=cmd_render_with_depth)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
