"""Real, end-to-end demonstration of `YT_IDV_MAX_INTENSITY_ALPHA` (see
`yt_idv_renderer.py`): a genuine Cycles render of a small marker sphere
(diameter `SPHERE_DIAMETER_KPC` -- real physical scale, default 35 kpc, see
below) sitting at the galaxy's own density peak / the domain's own center,
camera zoomed in close, volume rendered with the NEW value-driven-alpha
Maximum Intensity mode -- transparent at the low end of what's visible,
opaque at the high end -- so the diffuse outer halo fades away and the
dense core reads as solid, with the marker sphere giving a real sense of
physical scale relative to it.

First real run used a 10 kpc, bright yellow sphere and came back "I can't
quite see it" -- addressed three ways, all in `build_scene()`'s own
comments: (1) yellow sat too close to the "dusk" colormap's own warm
core color, so the sphere blended in instead of contrasting -- switched to
a saturated green; (2) `MIP_ALPHA_MAX` (`yt_idv_renderer.py`) was 1.0,
letting the volume's own near-camera material build up to FULL opacity in
front of the sphere, hiding it completely via Alpha-Over regardless of its
own real depth-clip visibility -- lowered to 0.7; (3) the sphere itself was
also made a few times bigger (10 -> 35 kpc), still tiny relative to the
~500 kpc frame. The camera-distance investigation below was done against
the ORIGINAL 10 kpc size and is kept as-is for the record -- it's about
where AMR block-boundary artifacts start dominating the frame, a question
that doesn't change with the marker's own size.

Same overall pipeline/subprocess-bridge structure as `render_real_demo.py`
-- see that script's own docstring for why the bridge exists at all
(yt_idv still can't import into Blender 5.3's bundled Python 3.13) -- this
one just uses different scene geometry and the new render mode.

Geometry: `IsolatedGalaxy`'s domain is ~1000 kpc wide (confirmed via
`ds.domain_width.in_units("kpc")`) and its density peak sits at unitary
(0.504, 0.500, 0.500) -- essentially the domain's own center (see
`render_partial_occlusion_demo.py`'s own docstring for how this was found).
With this script's usual box convention (edge length 2, world [-1, 1] <->
unitary [0, 1]), 10 kpc is `10/1000.1 * 2 ≈ 0.02` world units end to end --
genuinely tiny relative to the box.

**Zoom distance, tuned empirically, and why it's NOT close to the marker
sphere's own 10 kpc scale.** First instinct: put the camera a few multiples
of 10 kpc away (a genuine telephoto framing of the marker sphere itself).
Tried `CAMERA_DISTANCE` from 2 to 60 kpc (`sweep_camera_distance` below,
run against a fast synthetic camera first, then confirmed with real Cycles
renders at 15/30/50 kpc): every one of these showed a bizarre, sharp
TRIANGULAR silhouette instead of an organic gas cloud, and the per-view
auto-computed colormap range collapsed to a near-single value (`e.g.
4.4e-8` at both ends). Root cause: `yt_idv` renders the AMR hierarchy
block-by-block, each block a real triangulated cube mesh -- at 10-60 kpc
from the domain center, the camera is close enough to be effectively
pressed up against (or inside) a SINGLE grid block's own bounding cube, so
what's on screen is that one block's own flat rasterized face/corner, not
resolved gas structure (real density values scale color across that face,
but the SHAPE is geometric, not organic). Backing off to distances of
300-500 kpc (30-50x farther than the naive "a few sphere-diameters away"
instinct) lets the frame span MANY nested blocks at once, at which point
the AMR block edges blend smoothly and real structure (a compact bright
core, softer surrounding halo) becomes visible -- confirmed by comparing
renders at 300/400/500/650 kpc (`CAMERA_DISTANCE` 0.6/0.8/1.0/1.3): 500 kpc
(1.0) gave the cleanest result with no visible block-edge artifacts at all.
The marker sphere is still there at this distance -- genuinely tiny (a
handful of pixels at 512x512, confirmed directly against the real Cycles
beauty pass) -- which is itself the honest answer to "zoom in on a 10 kpc
sphere": at the distance where the SURROUNDING GAS STRUCTURE actually
resolves cleanly, a 10 kpc marker is necessarily a small dot, not a
frame-filling object; the two length scales (block granularity vs. desired
marker size) don't cooperate at this dataset's resolution.

Colormap/ramp tuning for `MAX_INTENSITY_ALPHA`'s own transparent-low/
opaque-high behavior (`MIP_ALPHA_COLORMAP`/`MAX`/`RAMP_POWER` in
`yt_idv_renderer.py`) is swept SEPARATELY and cheaply against this script's
own cached real-Cycles depth data, per the same "modify the transfer
function repeatedly and force a redraw in between snapshotting, no need to
reload/re-execute" technique already used for the threshold-fog tuning --
see `sweep_mip_alpha_params` below, run directly in the yt_idv venv against
`_zoomed_core_demo_work`'s cached `bridge_request.json`/`bridge_depth.npy`
from a real run of this script, NOT by re-running Cycles per candidate.
Swept 5 colormaps x 4 ramp powers (0.15/0.3/0.6/1.0): `MIP_ALPHA_RAMP_POWER`
lower than 1.0 keeps the diffuse halo faintly visible instead of vanishing
to fully transparent everywhere except the very core (mean alpha ranged
0.029 at power=1.0 up to 0.318 at power=0.15, same underlying data); 0.15
looked best. Colormap choice (tried dusk/viridis/arbre/inferno/algae) is a
matter of taste, not correctness -- `dusk` (blue halo -> orange/white core)
was picked for visual contrast between the transparent-low and opaque-high
ends, matching the transparency effect being demonstrated.
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
    "threshold_fog": "YT_IDV_THRESHOLD_FOG",
}

# See module docstring's "Geometry" section for the derivation.
DOMAIN_WIDTH_KPC = 1000.1  # IsolatedGalaxy's own domain_width.in_units("kpc")
SPHERE_DIAMETER_KPC = float(os.environ.get("SPHERE_DIAMETER_KPC", "35.0"))  # was 10.0 -- see build_scene()'s own comment on why this alone wasn't (and isn't) the main legibility fix
BOX_EDGE_WORLD = 2.0  # this project's usual convention: world [-1, 1] <-> unitary [0, 1]
SPHERE_RADIUS_WORLD = (SPHERE_DIAMETER_KPC / 2.0 / DOMAIN_WIDTH_KPC) * BOX_EDGE_WORLD

CAMERA_DISTANCE = float(os.environ.get("CAMERA_DISTANCE", "1.0"))  # world units from origin -- see module docstring's "Zoom distance, tuned empirically" section for why this is ~500 kpc, not something closer to the marker sphere's own 10 kpc scale
CAMERA_LENS_MM = float(os.environ.get("CAMERA_LENS_MM", "35.0"))

os.makedirs(WORK_DIR, exist_ok=True)
os.makedirs(IMAGES_DIR, exist_ok=True)
sys.path.insert(0, ADDON_SRC_PARENT)

from cycles_volume_override import external_render, operators, properties  # noqa: E402


def _bridge_render(request):
    """Drop-in replacement for a yt_idv_renderer.yt_idv_*_renderer function
    -- same RenderRequest -> RenderResult signature `operators.
    RENDERER_MODES` expects -- that shells out to the yt_idv/Python-3.12 venv
    instead of calling yt_idv in-process. See module docstring for why.
    """
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

    # The flagged "volume box": edge length 2 (spans -1..1 on each axis).
    # yt_idv_renderer's registration transform uses this object's own
    # bounding_box_local/matrix_world directly as the Blender-world <->
    # yt_idv-unitary mapping (see its module docstring, "Cycles Z-buffer
    # wiring") -- no special setup needed beyond flagging it.
    bpy.ops.mesh.primitive_cube_add(size=2.0, location=(0.0, 0.0, 0.0))
    box = bpy.context.active_object
    box.name = "FlaggedVolumeBox"
    box.volume_override.is_volume_override = True

    # Marker sphere at the box's own center -- matching the dataset's real
    # density peak (unitary (0.504, 0.500, 0.500), see module docstring),
    # radius = SPHERE_DIAMETER_KPC converted to this box's world units.
    #
    # First version of this demo used a small (10 kpc), bright YELLOW
    # sphere -- reported back as "I can't quite see it". Two real causes,
    # both fixed here, not just "make it bigger": (1) yellow sits too close
    # to the "dusk" colormap's own warm orange/white core color, so the
    # sphere visually blended into the gas around it instead of contrasting
    # against it -- switched to a saturated GREEN, a hue "dusk" never
    # produces. (2) `MIP_ALPHA_MAX` (yt_idv_renderer.py) was 1.0, meaning
    # the volume's own near-camera material could build up to FULL opacity
    # before the ray ever reached the sphere -- via Alpha-Over that hides
    # anything behind it completely, regardless of the sphere's own real
    # depth-clip visibility. Lowered to 0.7 so something behind the densest
    # visible gas always shows through. `SPHERE_DIAMETER_KPC` was also
    # bumped up (still tiny relative to the ~500 kpc frame, but a few times
    # larger) as a third, complementary fix -- see that constant's own
    # default.
    bpy.ops.mesh.primitive_uv_sphere_add(
        radius=SPHERE_RADIUS_WORLD, location=(0.0, 0.0, 0.0), segments=64, ring_count=32,
    )
    sphere = bpy.context.active_object
    sphere.name = "CoreMarkerSphere"

    mat = bpy.data.materials.new("CoreMarkerSphereMaterial")
    mat.use_nodes = True
    bsdf = mat.node_tree.nodes.get("Principled BSDF")
    if bsdf is not None:
        bsdf.inputs["Base Color"].default_value = (0.1, 1.0, 0.25, 1.0)  # saturated green -- see comment above
        if "Roughness" in bsdf.inputs:
            bsdf.inputs["Roughness"].default_value = 0.35
        if "Emission Color" in bsdf.inputs:
            bsdf.inputs["Emission Color"].default_value = (0.1, 1.0, 0.25, 1.0)
        if "Emission Strength" in bsdf.inputs:
            bsdf.inputs["Emission Strength"].default_value = 0.6  # raised from 0.15 -- needs to read clearly even partly behind translucent gas now
    sphere.data.materials.append(mat)

    # Camera: see module docstring's "Zoom distance" section for why this
    # is ~500 kpc (CAMERA_DISTANCE default 1.0 world unit), not a close-in
    # telephoto shot of the marker sphere itself -- closer distances put the
    # camera against/inside a single AMR block's own geometry instead of
    # resolving real gas structure.
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
    # Was 3.0 -- blew the specular highlight's linear R channel out past 1.0
    # (measured ~1.5) while G/B stayed under it, so this script's own
    # tonemap-free `_srgb_encode` clipped R first and left a warm/pink-white
    # highlight instead of a clean white one (see conversation: confirmed via
    # direct EXR inspection that the yt_idv volume layer was genuinely
    # (0,0,0,0) there -- this was a real Cycles specular highlight, not a
    # compositing artifact). Lowered so the highlight's linear peak stays
    # under 1.0 and all three channels clip together (or not at all).
    sun.data.energy = 1.2

    scene.render.engine = "CYCLES"
    scene.cycles.samples = 64
    scene.cycles.use_denoising = True
    scene.render.resolution_x = 512
    scene.render.resolution_y = 512  # square -- see yt_idv_renderer's own
    #                                   documented non-square-aspect caveat
    scene.render.film_transparent = True  # see module docstring, point 1
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
    """Write an (H, W, 4) float array straight to PNG via Blender's own
    image API (no external PNG library needed) -- caller is responsible for
    already having values in a sane 0..1-ish display range.
    """
    h, w = rgba_float.shape[:2]
    img = bpy.data.images.new("tmp_png_out", width=w, height=h, alpha=True)
    img.pixels.foreach_set(np.clip(rgba_float, 0.0, 1.0).astype("float32").ravel())
    img.filepath_raw = path
    img.file_format = "PNG"
    img.save()
    bpy.data.images.remove(img)


def _verify_and_export(scene):
    """Read pass 2's real final EXR back from disk (NOT the in-memory
    "Render Result" datablock -- in `--background` mode its `.pixels`
    buffer reads back as size (0, 0) after `write_still=True`, a background-
    mode quirk unrelated to whether the render itself is correct; the file
    Pass 2 actually wrote to disk is the authoritative output), verify the
    background is genuinely transparent (not opaque black), and write both
    a plain display-ready PNG and a checkerboard-composited PNG that makes
    transparency visually obvious.
    """
    final_path = scene.render.frame_path()
    if not os.path.exists(final_path):
        # Real Blender didn't append a frame-number suffix for this single-
        # frame still (frame_path()'s prediction and the file `render.render`
        # actually wrote disagreed by that suffix alone) -- fall back to
        # whatever actually landed on disk next to it.
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

    # Background-alpha check: a corner pixel should have neither the
    # sphere nor the box's volume silhouette anywhere near it.
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

    # Checkerboard-composited PNG: proves the alpha is really being used --
    # premultiplied-alpha-over onto a two-tone checker background. If the
    # background were actually opaque black, the checker would NOT show
    # through here.
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
    """Documents the technique used during development rather than being a
    maintained, always-runnable utility from THIS file: `main()`'s own
    `bpy`/`mathutils` imports at module level mean this file can only be
    imported inside real Blender, but the actual sweep needs to run in the
    SEPARATE yt_idv venv (a fast, pure-yt_idv hand-built camera aimed at the
    domain center from `candidates_kpc` away, no bpy/Cycles at all -- no
    `_registered_camera_kwargs`/real Blender camera needed for JUST this
    "where does the block-boundary triangulation artifact stop dominating
    the frame" question, see module docstring's "Zoom distance" section).
    Run as a standalone throwaway script in the yt_idv venv instead, loading
    `cycles_volume_override.external_render`/`yt_idv_renderer` via
    `importlib.util.spec_from_file_location` (the same pattern `ytidv_
    worker.py` uses) -- not by importing this file directly there.
    """
    raise NotImplementedError(
        f"see this function's own docstring -- run as a standalone script in "
        f"the yt_idv venv instead, sweeping candidates_kpc={candidates_kpc!r} "
        f"via a hand-built CameraParams/RenderRequest and yt_idv_max_intensity_"
        f"renderer(), not by calling this function directly."
    )


def sweep_mip_alpha_params(work_dir=WORK_DIR, colormaps=("dusk", "viridis", "arbre", "inferno", "algae"), ramp_powers=(0.15, 0.3, 0.6, 1.0)):
    """NOT run by `main()` -- run directly in the yt_idv venv AFTER a real
    run of this script has populated `work_dir` with `bridge_request.json`/
    `bridge_depth.npy` (i.e. run `main()` via real Blender first). Re-renders
    the SAME real Cycles request repeatedly with different `MIP_ALPHA_
    COLORMAP`/`RAMP_POWER` values -- no Cycles re-render, no dataset
    reload, just `yir.MIP_ALPHA_COLORMAP = ...; yir._mip_alpha_colormap_
    cache.clear(); yir.yt_idv_max_intensity_alpha_renderer(request)` per
    candidate (the "modify repeatedly, force a redraw in between
    snapshotting" technique) -- writes one PNG per (colormap, ramp_power)
    combination and prints each one's mean/max alpha.

    Must be run with THIS repo's `cycles_volume_override` package importable
    (e.g. via the same `importlib.util.spec_from_file_location` pattern
    `ytidv_worker.py` uses, since bpy isn't installed in the yt_idv venv and
    this file itself imports `bpy` at module level) -- not directly callable
    by importing this module in the yt_idv venv as-is.
    """
    import json

    import numpy as np
    import yt

    from cycles_volume_override import external_render, yt_idv_renderer as yir

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
        dataset_id="obj1", dataset_kind="yt", dataset_path=None, requested_time=0.0,
        time_units="FRAME_INDEX", mapping_policy="DIRECT", is_blender_volume_sequence=False,
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
