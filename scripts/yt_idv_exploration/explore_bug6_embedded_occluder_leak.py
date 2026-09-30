"""Checks whether an occluder embedded inside the domain leaks volume through
it, using an analytic ray/sphere depth buffer (no Blender or Cycles).

The leak seen in the real demo was caused by `_render_with_yt_idv` not calling
`scene.camera._update_matrices()` after `scene.camera.update(...)`; now fixed.
Kept for the record: nonzero leak counts after the fix may be legitimate
non-occlusion, since this test was never re-validated.
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

# Same geometry as render_real_demo.py: box [-1,1]^3, sphere radius 1 at corner (1,1,1).

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
        # Cycles depth is planar: project the hit onto `forward`.
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

    # Volume alpha should be 0 wherever the sphere occludes.
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

