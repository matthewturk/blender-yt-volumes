"""HISTORICAL -- kept for the record, no longer the recommended path. This
subprocess worker bridged this repo's own bpy==5.2.0/Python-3.13 dev venv
to a throwaway yt_idv/Python-3.12 venv (see ../README.md and
../setup_ytidv_venv.sh) to work around yt_idv's now-fixed unconditional
`imgui` dependency, which used to make it uninstallable under Python 3.13
at all. Fixed upstream 2026-09-15 (yt-project/yt_idv#268,
"imgui-optional") -- `bpy` and real `yt_idv` now coexist in ONE Python-3.13
venv (`uv sync --group ytidv` in this repo's own venv; see README
"Development & testing" -> "Headless yt_idv testing, in this same venv"),
so a script needing both no longer needs this bridge at all --
`render_real_demo.py` in this same directory was ported off it as the
worked example.

Original docstring, for context on why this worker exists:

Subprocess worker: the yt_idv side of a bridge between this repo's own
bpy==5.2.0/Python-3.13 dev venv and the throwaway yt_idv/Python-3.12 venv
(see ../README.md and ../setup_ytidv_venv.sh -- bpy and yt_idv couldn't be
installed in the same venv before the imgui fix above).

This is NOT part of the shipped add-on -- in real Blender, yt_idv_renderer.py
calls yt_idv in-process (see its module docstring), because real Blender's
own embedded Python doesn't have the bpy-as-a-pip-wheel version conflict
that forced this sandbox into two venvs. This worker exists purely so this
sandbox can exercise the real `yt_idv_max_intensity_renderer` function
driven by a REAL Blender camera (built with the add-on's own
`operators.build_camera_params()`) across multiple frames, which needs bpy
on one end and yt_idv on the other.

Run inside the ytidv venv (see ../setup_ytidv_venv.sh):

    describe_grids: dump the sample dataset's AMR grid bounding boxes (in
    yt_idv's own "unitary" [0, 1]^3 convention) as JSON, for the Blender
    side to build reference Empty objects from.

        python ytidv_worker.py describe_grids --out grids.json --max-grids 40

    render: load a JSON-serialized RenderRequest (built on the Blender side
    from a real Blender camera via operators.build_camera_params(), see
    render_frames_in_blender.py), call the real
    yt_idv_max_intensity_renderer(), save the result as a PNG.

        python ytidv_worker.py render --request request.json --out frame.png

    render_with_depth: like `render`, but takes a REAL per-pixel depth
    array (a .npy, float32, same row order named in the request JSON's
    "depth" block) instead of fabricating a uniform no_hit_value buffer --
    this is the one a real two-pass pipeline run actually needs, since the
    whole point of the Cycles Z-buffer wiring (see yt_idv_renderer.py's
    "Cycles Z-buffer wiring" docstring section) is a genuine per-pixel
    Z-pass, not a stand-in. Also takes --mode to pick which of the three
    real renderer functions to call, and writes the RGBA result as a full-
    float .npy (not a PNG) so premultiplied alpha survives the round trip
    for the real add-on compositor to consume -- see render_real_demo.py.

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
ADDON_SRC = os.path.normpath(os.path.join(HERE, "..", "..", "..", "src", "cycles_volume_override"))


def _load_addon_module(name, path):
    """Load one add-on submodule directly, bypassing
    cycles_volume_override/__init__.py's eager bpy-dependent imports (no
    bpy in this venv) -- see module docstring.
    """
    spec = importlib.util.spec_from_file_location(f"cycles_volume_override.{name}", path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod


def _addon_modules():
    pkg = types.ModuleType("cycles_volume_override")
    pkg.__path__ = [ADDON_SRC]
    sys.modules.setdefault("cycles_volume_override", pkg)
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
        requested_time=0.0, time_units="FRAME_INDEX", mapping_policy="DIRECT",
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
    """Render every request in a JSON list in ONE process -- unlike calling
    `render` once per frame (a fresh subprocess each time, which would pay
    yt_idv_renderer's dataset-load + GL-context-creation cost on every
    single frame), this actually exercises `_load_sample_source`/
    `_get_context`'s module-level caching the way a real, long-running
    Blender session would: first frame pays the setup cost, every frame
    after reuses it. See render_frames_in_blender.py for the driver that
    builds `requests.json` from a real, moving Blender camera.
    """
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


_REAL_RENDER_MODES = None  # populated lazily -- see cmd_render_with_depth


def _build_request_full(external_render, req, depth_array):
    """Like `_build_request` above, but takes REAL camera/volume/depth
    values straight from a real Blender scene (see render_real_demo.py)
    instead of hand-rolled test fixtures. `req["camera"]["projection_matrix"]`
    in particular is expected to be the actual
    `camera_obj.calc_matrix_camera(...)` result -- see operators.
    build_camera_params() -- not the standalone OpenGL-formula stand-in
    `explore_cycles_zbuffer_depth_clip.py` uses.
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
        requested_time=0.0, time_units="FRAME_INDEX", mapping_policy="DIRECT",
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
        "threshold_fog": yt_idv_renderer.yt_idv_threshold_fog_renderer,
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
    p_render_depth.add_argument("--mode", choices=["max_intensity", "max_intensity_alpha", "transfer_function", "projection", "threshold_fog"], default="projection")
    p_render_depth.add_argument("--out-npy", required=True)
    p_render_depth.set_defaults(func=cmd_render_with_depth)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
