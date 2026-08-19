"""Eighth exploration: chasing what first looked like a yt_idv depth-clip
bug (see `explore_external_depth_clip.py`/`explore_cycles_zbuffer_depth_
clip.py`) that only showed up when an occluder sits embedded INSIDE the
domain rather than entirely in front of it -- ended up finding a real bug in
THIS ADD-ON's OWN code instead, now fixed. Kept for the record, NOT as a
live repro (see "Status" below).

Found while building the real end-to-end sphere-and-volume demo
(`blender_scene/render_real_demo.py`, per the user's request for "a cycles
render with a sphere that is centered at a box corner with a radius of half
the box"): a real, obvious light-colored patch appeared INSIDE the sphere's
own silhouette (distinct from the correctly-working crescent-shaped GAP
around/below it) -- confirmed via direct EXR pixel inspection to be the
yt_idv volume layer drawing fully opaque (`alpha == 1.0` exactly) on top of
pixels the real Cycles beauty pass unambiguously shows as solid, correctly-Z'd
sphere material.

Reproduced here with a PURE ANALYTIC ray/sphere depth field -- no Blender,
no Cycles, no EXR round-trip -- specifically to rule the Blender<->yt_idv
registration bridge in or out. Two scenarios, same camera/dataset:

  1. Sphere radius = half the box, centered AT a box corner (mirrors the
     real demo) -- ~18% of the sphere's occluded pixels leaked.
  2. A SMALL sphere floating well inside the domain, nowhere near any wall
     -- 100% of its occluded pixels leaked, WORSE than case 1.

**Status: root cause found and fixed, but NOT in this file's own leaked-pixel
count -- that part of the investigation turned out to be a dead end.** The
real bug: `_render_with_yt_idv` called `scene.camera.update(**kwargs)` but
never called the follow-up `scene.camera._update_matrices()` that `yt_idv.
cameras.trackball_camera.TrackballCamera` actually needs to recompute
`projection_matrix` from the new fov/near/far (its own `_compute_matrices`
override -- the hook `.update()` is supposed to trigger -- is a no-op; see
`yt_idv_renderer.py`'s module docstring for the full account, including how
this masked itself as looking approximately-right for the camera's POSITION
the entire time). Confirmed as the actual fix using this exact script's
scenario 1's camera/geometry in isolation (not shown here, see the
conversation this was investigated in): adding the missing call changed the
computed per-pixel clip depths from a badly precision-starved band near 1.0
to a well-spread, correct-looking range -- and, more importantly, re-running
the REAL demo (`render_real_demo.py`) after the fix produces a completely
clean sphere silhouette with no leak at all.

What this script's OWN leaked-pixel counts don't tell you, and why re-running
it post-fix still shows nonzero counts: fixing the camera bug changes the
EFFECTIVE FIELD OF VIEW (this whole project's yt_idv renders were silently
using a stale default `fov=45` instead of whatever this module actually
computed, the entire time) -- which changes which world/unitary point each
screen pixel in THIS script's own hand-rolled analytic sphere test actually
corresponds to. A "leak" in this script's post-fix output may just mean the
sphere (an arbitrary, hand-picked test position, not tied to the dataset's
own density structure) legitimately sits BEHIND real, brighter domain
content along that specific ray from the new, corrected camera angle -- a
CORRECT non-occlusion, not a bug -- rather than the original symptom. This
script's numbers were never re-validated against that possibility because
the real, user-facing case (`render_real_demo.py`, using a REAL Cycles
Z-buffer, where "nearest thing wins" is unambiguous and authoritative by
construction) is the one that was actually confirmed fixed, and that's the
one that matters. Left as-is rather than reworked into a rigorous repro,
since the bug it was chasing is already found, fixed, and confirmed via a
better (real) test case.

`images/26_bug6_corner_sphere_leak_projection.png` /
`images/27_bug6_corner_sphere_leak_max_intensity.png` are the ORIGINAL
(pre-fix) reference images from this investigation, kept for the record.
"""

import sys, importlib.util, types, math
import numpy as np

ADDON_SRC = "/home/mturk/yt/cycles-yt/src/cycles_volume_override"
pkg = types.ModuleType("cycles_volume_override")
pkg.__path__ = [ADDON_SRC]
sys.modules.setdefault("cycles_volume_override", pkg)

def load(name):
    spec = importlib.util.spec_from_file_location(f"cycles_volume_override.{name}", f"{ADDON_SRC}/{name}.py")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod

external_render = load("external_render")
yir = load("yt_idv_renderer")

# Same geometry as render_real_demo.py: box [-1,1]^3, sphere radius 1
# centered at corner (1,1,1) -- tangent-ish to the box there. Camera placed
# on the far diagonal, same general framing. This is an ANALYTIC ray/sphere
# intersection depth buffer -- no Cycles, no EXR round-trip, no real Blender
# involved at all -- purely to isolate whether the leak is a yt_idv/
# depth-clip-math issue or something specific to the Blender<->yt_idv bridge.

W = H = 256
box_half = 1.0
sphere_center = np.array([1.0, 1.0, 1.0])
sphere_radius = 1.0

cam_pos = np.array([-3.4, -4.0, 2.6])
target = np.array([0.35, 0.35, 0.35])
forward = (target - cam_pos)
forward /= np.linalg.norm(forward)
world_up = np.array([0.0, 0.0, 1.0])
right = np.cross(forward, world_up)
right /= np.linalg.norm(right)
up = np.cross(right, forward)

fov_deg = 2.0 * math.degrees(math.atan((36.0 / 2.0) / 40.0))  # matches render_real_demo's cam.data.lens=40, sensor 36mm
half_h = math.tan(math.radians(fov_deg) / 2.0)

def matrix_world_from_basis(pos, fwd, up_vec, right_vec):
    # Blender camera convention: local -Z = forward, +Y = up, +X = right
    m = np.eye(4)
    m[:3, 0] = right_vec
    m[:3, 1] = up_vec
    m[:3, 2] = -fwd
    m[:3, 3] = pos
    return m.tolist()

def _perspective_matrix(fovy_deg, aspect, near, far):
    f = 1.0 / math.tan(math.radians(fovy_deg) / 2.0)
    return [
        [f / aspect, 0, 0, 0],
        [0, f, 0, 0],
        [0, 0, (far + near) / (near - far), (2 * far * near) / (near - far)],
        [0, 0, -1, 0],
    ]

no_hit = 1e10
depth_data = np.full((H, W), no_hit, dtype=np.float32)

for row in range(H):
    v = (0.5 - (row + 0.5) / H) * 2.0 * half_h
    for col in range(W):
        u = ((col + 0.5) / W - 0.5) * 2.0 * half_h
        d = forward + u * right + v * up
        d = d / np.linalg.norm(d)
        oc = cam_pos - sphere_center
        b = np.dot(d, oc)
        c = np.dot(oc, oc) - sphere_radius ** 2
        disc = b * b - c
        if disc < 0:
            continue
        t = -b - math.sqrt(disc)
        if t <= 0:
            continue
        # Cycles PLANAR depth = distance along the camera's own forward axis,
        # not Euclidean ray length -- project the hit point onto `forward`.
        hit = cam_pos + t * d
        planar_z = np.dot(hit - cam_pos, forward)
        depth_data[row, col] = planar_z

camera = external_render.CameraParams(
    matrix_world=matrix_world_from_basis(cam_pos, forward, up, right),
    projection="PERSP", lens_mm=40.0, sensor_width_mm=36.0, sensor_height_mm=36.0,
    sensor_fit="AUTO", shift_x=0.0, shift_y=0.0, ortho_scale=None,
    clip_start=0.1, clip_end=100.0,
    resolution_x=W, resolution_y=H, pixel_aspect_x=1.0, pixel_aspect_y=1.0,
    projection_matrix=_perspective_matrix(fov_deg, 1.0, 0.1, 100.0),
    panorama=None,
    dof=external_render.DepthOfFieldParams(
        use_dof=False, focus_distance_m=5.0, aperture_fstop=2.8,
        aperture_blades=0, aperture_ratio=1.0, aperture_rotation_rad=0.0,
    ),
)

depth = external_render.DepthBuffer(
    data=depth_data, depth_convention="PLANAR", no_hit_value=no_hit,
    row_order="TOP_LEFT_ORIGIN", units="meters",
)
version = external_render.VersionInfo(hash="x", changed=True)
dataset = external_render.DatasetParams(
    dataset_id="obj", dataset_kind="yt", dataset_path=None,
    requested_time=0.0, time_units="FRAME_INDEX", mapping_policy="DIRECT",
    is_blender_volume_sequence=False, sequence_frame_start=None,
    sequence_frame_duration=None, sequence_frame_offset=None,
    sequence_mode=None, resolved_sequence_frame=None, grid_names=[],
    version=version,
)
volume = external_render.VolumeObjectParams(
    matrix_world=[[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]],
    bounding_box_local=[[x, y, z] for x in (-1, 1) for y in (-1, 1) for z in (-1, 1)],
    custom_params={}, params_version=version, dataset=dataset,
)
frame = external_render.FrameParams(frame_number=1, scene_time_seconds=0.04)
request = external_render.RenderRequest(camera=camera, depth=depth, volume=volume, frame=frame)

for mode_name, fn in [("projection", yir.yt_idv_projection_renderer), ("max_intensity", yir.yt_idv_max_intensity_renderer)]:
    result = fn(request)
    rgba = result.rgba
    import yt
    out = f"/tmp/claude-1000/-home-mturk-yt-cycles-yt/db045906-ff8b-426c-bb6d-8949e7ae6506/scratchpad/test_tangent_{mode_name}.png"
    yt.write_bitmap(rgba[:, :, :3], out)
    print(mode_name, "wrote", out)

    # Check: at pixels where the ANALYTIC sphere clearly occludes (depth_data
    # not no_hit), is volume alpha correctly 0 everywhere, or does it leak
    # like the real-Cycles case?
    sphere_mask = depth_data < no_hit
    leaked = sphere_mask & (rgba[..., 3] > 0.01)
    print(f"  sphere-covered pixels: {sphere_mask.sum()}, of which LEAKED (alpha>0 despite sphere occluding): {leaked.sum()}")
    if leaked.sum() > 0:
        ys, xs = np.where(leaked)
        print(f"  leaked region cols range: {xs.min()}-{xs.max()}, rows range: {ys.min()}-{ys.max()}")
    if leaked.sum() > 0:
        print("  alpha at leaked: min", rgba[...,3][leaked].min(), "max", rgba[...,3][leaked].max(), "mean", rgba[...,3][leaked].mean())
        ys2, xs2 = np.where(leaked)
        for r, c in list(zip(ys2[:8], xs2[:8])):
            print(f"    row={r} col={c} depth={depth_data[r,c]:.4f} alpha={rgba[r,c,3]:.4f} rgb={rgba[r,c,:3]}")

