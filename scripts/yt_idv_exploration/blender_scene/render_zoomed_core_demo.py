"""Demo of `YT_IDV_MAX_INTENSITY_ALPHA`: a small Cycles marker sphere
(`SPHERE_DIAMETER_KPC`) at IsolatedGalaxy's density peak (the domain
center), with value-driven alpha so the diffuse halo fades and the dense
core reads as solid.

Run with real Blender and the yt_idv subprocess bridge (`ytidv_worker.py`,
venv from YTIDV_VENV_PYTHON):

    /data/SteamLibrary/steamapps/common/Blender/blender --background \\
        --python scripts/yt_idv_exploration/blender_scene/render_zoomed_core_demo.py

Env knobs: VOLUME_OVERRIDE_MODE, SPHERE_DIAMETER_KPC, CAMERA_DISTANCE,
CAMERA_LENS_MM.

Scale: the domain is ~1000 kpc wide and maps onto a box spanning world
[-1, 1], so CAMERA_DISTANCE=1.0 is ~500 kpc from the center.

Gotcha: with the camera much closer (tens of kpc), it sits against or
inside a single AMR block and the frame shows that block's flat, triangular
geometry instead of gas structure. ~500 kpc resolves cleanly, at the cost of
the marker being only a few pixels across.

`MIP_ALPHA_*` in `yt_idv_renderer.py` controls the ramp; see
`sweep_mip_alpha_params` to tune it against cached depth data without
re-running Cycles.
"""

import json
import os
import subprocess
import sys

import bpy
import mathutils
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.normpath(os.path.join(HERE, "..", "..", ".."))
ADDON_SRC_PARENT = os.path.join(REPO_ROOT, "src")
IMAGES_DIR = os.path.normpath(os.path.join(HERE, "..", "images"))

YTIDV_VENV_PYTHON = os.environ.get(
    "YTIDV_VENV_PYTHON",
    "/tmp/claude-1000/-home-mturk-yt-cycles-yt/db045906-ff8b-426c-bb6d-8949e7ae6506/scratchpad/ytidv_venv/bin/python",
)
WORKER_SCRIPT = os.path.join(HERE, "ytidv_worker.py")
WORK_DIR = os.path.join(HERE, "_zoomed_core_demo_work")

RENDERER_MODE = os.environ.get("VOLUME_OVERRIDE_MODE", "max_intensity_alpha")
MODE_KEY_BY_NAME = {
    "max_intensity": "YT_IDV_MAX_INTENSITY",
    "max_intensity_alpha": "YT_IDV_MAX_INTENSITY_ALPHA",
    "transfer_function": "YT_IDV_TRANSFER_FUNCTION",
    "projection": "YT_IDV_PROJECTION",
}

DOMAIN_WIDTH_KPC = 1000.1  # IsolatedGalaxy domain_width.in_units("kpc")
SPHERE_DIAMETER_KPC = float(os.environ.get("SPHERE_DIAMETER_KPC", "35.0"))
BOX_EDGE_WORLD = 2.0  # world [-1, 1] <-> unitary [0, 1]
SPHERE_RADIUS_WORLD = (SPHERE_DIAMETER_KPC / 2.0 / DOMAIN_WIDTH_KPC) * BOX_EDGE_WORLD

CAMERA_DISTANCE = float(os.environ.get("CAMERA_DISTANCE", "1.0"))  # world units from origin (1.0 ~ 500 kpc)
CAMERA_LENS_MM = float(os.environ.get("CAMERA_LENS_MM", "35.0"))

os.makedirs(WORK_DIR, exist_ok=True)
os.makedirs(IMAGES_DIR, exist_ok=True)
sys.path.insert(0, ADDON_SRC_PARENT)

from yt_volumes import external_render, operators, properties  # noqa: E402


def _bridge_render(request):
    """Stand-in for a yt_idv_*_renderer that shells out to the yt_idv venv."""
    req_json = {
        "camera": {
            "matrix_world": request.camera.matrix_world,
            "projection": request.camera.projection,
            "lens_mm": request.camera.lens_mm,
            "sensor_width_mm": request.camera.sensor_width_mm,
            "sensor_height_mm": request.camera.sensor_height_mm,
            "sensor_fit": request.camera.sensor_fit,
            "shift_x": request.camera.shift_x,
            "shift_y": request.camera.shift_y,
            "ortho_scale": request.camera.ortho_scale,
            "clip_start": request.camera.clip_start,
            "clip_end": request.camera.clip_end,
            "resolution_x": request.camera.resolution_x,
            "resolution_y": request.camera.resolution_y,
            "pixel_aspect_x": request.camera.pixel_aspect_x,
            "pixel_aspect_y": request.camera.pixel_aspect_y,
            "projection_matrix": request.camera.projection_matrix,
        },
        "depth": {
            "depth_convention": request.depth.depth_convention,
            "no_hit_value": request.depth.no_hit_value,
            "row_order": request.depth.row_order,
            "units": request.depth.units,
        },
        "volume": {
            "matrix_world": request.volume.matrix_world,
            "bounding_box_local": request.volume.bounding_box_local,
        },
        "frame_number": request.frame.frame_number,
    }
    req_path = os.path.join(WORK_DIR, "bridge_request.json")
    depth_path = os.path.join(WORK_DIR, "bridge_depth.npy")
    out_path = os.path.join(WORK_DIR, "bridge_result.npy")
    with open(req_path, "w") as f:
        json.dump(req_json, f)
    np.save(depth_path, np.asarray(request.depth.data, dtype=np.float32))

    subprocess.run(
        [
            YTIDV_VENV_PYTHON, WORKER_SCRIPT, "render_with_depth",
            "--request", req_path, "--depth-npy", depth_path,
            "--mode", RENDERER_MODE, "--out-npy", out_path,
        ],
        check=True, cwd=HERE,
    )
    rgba = np.load(out_path)
    return external_render.RenderResult(rgba=rgba, alpha_mode="PREMULTIPLIED", color_space="LINEAR")


class _StubPrefs:
    renderer_mode = MODE_KEY_BY_NAME[RENDERER_MODE]
    work_dir = WORK_DIR
    verbose_logging = True


def build_scene():
    scene = bpy.context.scene
    bpy.ops.object.select_all(action="SELECT")
    bpy.ops.object.delete()

    # Flagged volume box spanning -1..1; its bounds map to yt_idv's domain.
    bpy.ops.mesh.primitive_cube_add(size=2.0, location=(0.0, 0.0, 0.0))
    box = bpy.context.active_object
    box.name = "FlaggedVolumeBox"
    box.volume_override.is_volume_override = True

    # Marker sphere at the density peak (box center). Green contrasts with
    # the "dusk" colormap; MIP_ALPHA_MAX < 1 keeps dense gas in front of it
    # from hiding it completely.
    bpy.ops.mesh.primitive_uv_sphere_add(
        radius=SPHERE_RADIUS_WORLD, location=(0.0, 0.0, 0.0), segments=64, ring_count=32,
    )
    sphere = bpy.context.active_object
    sphere.name = "CoreMarkerSphere"

    mat = bpy.data.materials.new("CoreMarkerSphereMaterial")
    mat.use_nodes = True
    bsdf = mat.node_tree.nodes.get("Principled BSDF")
    if bsdf is not None:
        bsdf.inputs["Base Color"].default_value = (0.1, 1.0, 0.25, 1.0)
        if "Roughness" in bsdf.inputs:
            bsdf.inputs["Roughness"].default_value = 0.35
        if "Emission Color" in bsdf.inputs:
            bsdf.inputs["Emission Color"].default_value = (0.1, 1.0, 0.25, 1.0)
        if "Emission Strength" in bsdf.inputs:
            bsdf.inputs["Emission Strength"].default_value = 0.6  # reads through translucent gas
    sphere.data.materials.append(mat)

    # Camera distance: see the module docstring's AMR-block gotcha.
    cam_pos = mathutils.Vector((-0.7, -0.8, 0.55)).normalized() * CAMERA_DISTANCE
    bpy.ops.object.camera_add(location=cam_pos)
    cam = bpy.context.active_object
    cam.name = "DemoCamera"
    target = mathutils.Vector((0.0, 0.0, 0.0))
    direction = target - cam.location
    cam.rotation_euler = direction.to_track_quat("-Z", "Y").to_euler()
    cam.data.lens = CAMERA_LENS_MM
    scene.camera = cam

    bpy.ops.object.light_add(type="SUN", location=(4.0, -3.0, 6.0))
    sun = bpy.context.active_object
    # Kept low so the specular highlight stays under 1.0 and doesn't clip to
    # pink in the tonemap-free `_srgb_encode`.
    sun.data.energy = 1.2

    scene.render.engine = "CYCLES"
    scene.cycles.samples = 64
    scene.cycles.use_denoising = True
    scene.render.resolution_x = 512
    scene.render.resolution_y = 512
    scene.render.film_transparent = True
    scene.render.image_settings.color_mode = "RGBA"
    scene.render.image_settings.file_format = "OPEN_EXR"
    scene.render.image_settings.color_depth = "32"
    scene.render.filepath = os.path.join(WORK_DIR, "final_output_")

    return box, sphere, cam


def _srgb_encode(linear):
    linear = np.clip(linear, 0.0, None)
    return np.where(
        linear <= 0.0031308,
        linear * 12.92,
        1.055 * np.power(linear, 1.0 / 2.4) - 0.055,
    )


def _write_png(path, rgba_float):
    """Write an (H, W, 4) float array (display range 0..1) to PNG via bpy."""
    h, w = rgba_float.shape[:2]
    img = bpy.data.images.new("tmp_png_out", width=w, height=h, alpha=True)
    img.pixels.foreach_set(np.clip(rgba_float, 0.0, 1.0).astype("float32").ravel())
    img.filepath_raw = path
    img.file_format = "PNG"
    img.save()
    bpy.data.images.remove(img)


def _verify_and_export(scene):
    """Read pass 2's final EXR, check background alpha, and write PNGs.

    Reads the file on disk: headless, the "Render Result" image's `.pixels`
    is empty after `write_still=True`.
    """
    final_path = scene.render.frame_path()
    if not os.path.exists(final_path):
        # frame_path() may predict a frame-number suffix that wasn't written.
        import glob
        candidates = sorted(glob.glob(scene.render.filepath + "*"))
        if not candidates:
            raise RuntimeError(f"no final output found near {scene.render.filepath!r} (predicted {final_path!r})")
        final_path = candidates[-1]
    img = bpy.data.images.load(final_path, check_existing=False)
    try:
        w, h = img.size[0], img.size[1]
        pixels = np.array(img.pixels[:], dtype=np.float32).reshape(h, w, img.channels)
    finally:
        bpy.data.images.remove(img)

    # A corner pixel should be empty background.
    corner_alpha = float(pixels[2, 2, 3])
    print(f"[render_zoomed_core_demo] background corner pixel alpha = {corner_alpha:.4f} "
          f"(expect ~0.0)")
    if corner_alpha > 0.02:
        print("[render_zoomed_core_demo] WARNING: background does not look transparent -- "
              "film_transparent or the yt_idv layer's own alpha may not be wired correctly")

    # Plain display-ready PNG: sRGB-encode the premultiplied RGB, keep alpha.
    display = pixels.copy()
    display[..., :3] = _srgb_encode(display[..., :3])
    plain_path = os.path.join(IMAGES_DIR, "32_zoomed_core_sphere_and_volume.png")
    _write_png(plain_path, display)
    print(f"[render_zoomed_core_demo] wrote {plain_path}")

    # Premultiplied alpha-over onto a checkerboard, to show transparency.
    square = 32
    yy, xx = np.mgrid[0:h, 0:w]
    checker = (((yy // square) + (xx // square)) % 2).astype(np.float32)
    checker_rgb = 0.25 + 0.35 * checker[..., None]  # two mid-grays
    composited_rgb = display[..., :3] + checker_rgb * (1.0 - display[..., 3:4])
    composited = np.concatenate([composited_rgb, np.ones_like(display[..., 3:4])], axis=-1)
    checker_path = os.path.join(IMAGES_DIR, "33_zoomed_core_over_checkerboard.png")
    _write_png(checker_path, composited)
    print(f"[render_zoomed_core_demo] wrote {checker_path}")


def main():
    properties.register()
    operators.RENDERER_MODES[MODE_KEY_BY_NAME[RENDERER_MODE]] = _bridge_render
    operators._get_prefs = lambda context: _StubPrefs()

    box, sphere, cam = build_scene()
    print(f"[render_zoomed_core_demo] scene built: box={box.name} sphere={sphere.name} "
          f"cam={cam.name} mode={RENDERER_MODE} sphere_radius_world={SPHERE_RADIUS_WORLD:.5f} "
          f"camera_distance={CAMERA_DISTANCE} camera_lens_mm={CAMERA_LENS_MM}")

    operators.render_current_frame(bpy.context)
    print(f"[render_zoomed_core_demo] real two-pass pipeline finished; "
          f"final EXR at {bpy.context.scene.render.frame_path()!r}")

    _verify_and_export(bpy.context.scene)


def sweep_camera_distance(candidates_kpc=(2.0, 5.0, 10.0, 20.0, 40.0, 60.0, 95.0)):
    """Placeholder documenting the camera-distance sweep; not runnable here.

    Do it as a standalone script in the yt_idv venv: a hand-built camera at
    each distance, rendered with yt_idv_max_intensity_renderer(), loading the
    add-on modules the way `ytidv_worker.py` does.
    """
    raise NotImplementedError(
        f"see this function's own docstring -- run as a standalone script in "
        f"the yt_idv venv instead, sweeping candidates_kpc={candidates_kpc!r} "
        f"via a hand-built CameraParams/RenderRequest and yt_idv_max_intensity_"
        f"renderer(), not by calling this function directly."
    )


def sweep_mip_alpha_params(work_dir=WORK_DIR, colormaps=("dusk", "viridis", "arbre", "inferno", "algae"), ramp_powers=(0.15, 0.3, 0.6, 1.0)):
    """Re-render the cached request over MIP_ALPHA colormaps/ramp powers.

    Not run by `main()`. Run in the yt_idv venv after a real run has written
    `bridge_request.json`/`bridge_depth.npy` to `work_dir`, with
    `yt_volumes` importable (as in `ytidv_worker.py`; this file
    imports bpy at module level). Writes one PNG per combination.
    """
    import json

    import numpy as np
    import yt

    from yt_volumes import external_render, yt_idv_renderer as yir

    with open(os.path.join(work_dir, "bridge_request.json")) as f:
        req = json.load(f)
    depth_data = np.load(os.path.join(work_dir, "bridge_depth.npy"))

    camera = external_render.CameraParams(
        matrix_world=req["camera"]["matrix_world"], projection=req["camera"]["projection"],
        lens_mm=req["camera"]["lens_mm"], sensor_width_mm=req["camera"]["sensor_width_mm"],
        sensor_height_mm=req["camera"]["sensor_height_mm"], sensor_fit=req["camera"]["sensor_fit"],
        shift_x=req["camera"]["shift_x"], shift_y=req["camera"]["shift_y"], ortho_scale=req["camera"]["ortho_scale"],
        clip_start=req["camera"]["clip_start"], clip_end=req["camera"]["clip_end"],
        resolution_x=req["camera"]["resolution_x"], resolution_y=req["camera"]["resolution_y"],
        pixel_aspect_x=req["camera"]["pixel_aspect_x"], pixel_aspect_y=req["camera"]["pixel_aspect_y"],
        projection_matrix=req["camera"]["projection_matrix"], panorama=None,
        dof=external_render.DepthOfFieldParams(
            use_dof=False, focus_distance_m=5.0, aperture_fstop=2.8,
            aperture_blades=0, aperture_ratio=1.0, aperture_rotation_rad=0.0,
        ),
    )
    depth = external_render.DepthBuffer(
        data=depth_data, depth_convention=req["depth"]["depth_convention"],
        no_hit_value=req["depth"]["no_hit_value"], row_order=req["depth"]["row_order"], units=req["depth"]["units"],
    )
    version = external_render.VersionInfo(hash="x", changed=True)
    dataset = external_render.DatasetParams(
        dataset_id="obj1", dataset_kind="yt", dataset_path=None,
        is_blender_volume_sequence=False,
        sequence_frame_start=None, sequence_frame_duration=None, sequence_frame_offset=None,
        sequence_mode=None, resolved_sequence_frame=None, grid_names=[], version=version,
    )
    volume = external_render.VolumeObjectParams(
        matrix_world=req["volume"]["matrix_world"], bounding_box_local=req["volume"]["bounding_box_local"],
        custom_params={}, params_version=version, dataset=dataset,
    )
    frame = external_render.FrameParams(frame_number=req["frame_number"], scene_time_seconds=req["frame_number"] / 24.0)
    request = external_render.RenderRequest(camera=camera, depth=depth, volume=volume, frame=frame)

    for colormap in colormaps:
        for ramp_power in ramp_powers:
            yir.MIP_ALPHA_COLORMAP = colormap
            yir.MIP_ALPHA_RAMP_POWER = ramp_power
            yir._mip_alpha_colormap_cache.clear()
            result = yir.yt_idv_max_intensity_alpha_renderer(request)
            rgba = result.rgba
            out = os.path.join(IMAGES_DIR, f"_sweep_mipalpha_{colormap}_{ramp_power}.png")
            yt.write_bitmap(rgba[:, :, :3], out)
            print(f"{colormap} ramp={ramp_power}: alpha mean={rgba[..., 3].mean():.4f} max={rgba[..., 3].max():.4f} -> {out}")


if __name__ == "__main__":
    main()
