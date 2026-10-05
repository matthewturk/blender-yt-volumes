"""Demonstrates the external depth clip added to yt_idv's ray-marching shader
(`external_depth_tex`/`use_external_depth_clip`; see
`yt_idv_external_depth_clip.patch`), which stops rays at a per-pixel depth.

Run inside the throwaway venv (see setup_ytidv_venv.sh), against the patched
yt_idv checkout.

Gotchas:
  - Import `yt_idv.opengl_support` only after `yt_idv.render_context(...)`,
    which sets PYOPENGL_PLATFORM; importing earlier leaves newer GL functions
    (e.g. `glTexStorage3D`) undefined.
  - The `viewport` uniform is raw `(x, y, width, height)` pixels, not the
    `1 / screen` documented in `known_uniforms.inc.glsl`; divide by `.zw`.
  - `Texture2D` needs `depth[::-1, :].T` for a top-down (H, W) array; this is
    handled by `yt_idv_renderer.set_external_depth_clip`.

Renders a uniform depth wall at decreasing distances and a sphere-shaped
per-pixel clip, and checks that disabling the clip restores the baseline.
"""

import numpy as np
import yt
import yt_idv

ds = yt.load_sample("IsolatedGalaxy")
dd = ds.all_data()

W, H = 256, 256
rc = yt_idv.render_context("egl", width=W, height=H)
from yt_idv.opengl_support import Texture2D  # noqa: E402 -- see module docstring's import-order gotcha

scene = rc.add_scene(dd, "density", no_ghost=True)
comp = scene.components[0]


def window_depth_at_distance(cam, dist):
    """Window-space depth (0..1) of a point `dist` along the camera's view direction."""
    forward = np.array(cam.focus) - np.array(cam.position)
    forward = forward / np.linalg.norm(forward)
    world_pt = np.array(cam.position) + forward * dist
    clip = cam.projection_matrix @ (cam.view_matrix @ np.array([*world_pt, 1.0]))
    ndc_z = clip[2] / clip[3]
    return 0.5 * ndc_z + 0.5


def verify_axis_convention():
    """Check that `depth[::-1, :].T` puts a top-left clip in the top-left of the render."""
    comp.render_method = "max_intensity"
    scene.camera.update(position=[1.6, 1.1, 2.0], focus=[0.5, 0.5, 0.5], up=[0, 1, 0])
    depth = np.ones((H, W), dtype="f4")
    depth[0:96, 0:96] = 0.0  # top-left quadrant, ordinary top-down convention
    texture_data = depth[::-1, :].T
    comp.external_depth_texture = Texture2D(data=texture_data)
    comp.use_external_depth_clip = True
    result = rc.run()
    yt.write_bitmap(result[:, :, :3], "axisfix_topleft_test.png")
    comp.use_external_depth_clip = False


def uniform_wall_test():
    comp.render_method = "max_intensity"
    scene.camera.update(position=[1.6, 1.1, 2.0], focus=[0.5, 0.5, 0.5], up=[0, 1, 0])
    baseline = rc.run()
    yt.write_bitmap(baseline[:, :, :3], "16_depthclip_baseline.png")

    dist_to_focus = np.linalg.norm(np.array(scene.camera.focus) - np.array(scene.camera.position))

    comp.external_depth_texture = Texture2D(
        data=np.full((H, W), window_depth_at_distance(scene.camera, dist_to_focus * 0.6), dtype="f4")
    )
    comp.use_external_depth_clip = True
    clipped_60 = rc.run()
    print("uniform wall @0.6x differs from baseline:", not np.allclose(baseline, clipped_60))
    yt.write_bitmap(clipped_60[:, :, :3], "17_depthclip_clipped_0.6x.png")

    comp.external_depth_texture = Texture2D(
        data=np.full((H, W), window_depth_at_distance(scene.camera, dist_to_focus * 0.3), dtype="f4")
    )
    clipped_30 = rc.run()
    yt.write_bitmap(clipped_30[:, :, :3], "18_depthclip_clipped_0.3x.png")

    comp.use_external_depth_clip = False
    restored = rc.run()
    print("disabling clip exactly reproduces baseline:", np.allclose(baseline, restored))


def sphere_shaped_clip_test():
    """Clip with a sphere-shaped per-pixel depth that partly overlaps the gas."""
    comp.render_method = "transfer_function"
    comp.tf_min, comp.tf_max, comp.tf_log = 1e-6, 2e-3, True
    tf = yt.ColorTransferFunction((np.log(1e-6), np.log(2e-3)))
    tf.add_layers(5, colormap="viridis")
    rgba = np.stack([f.y for f in tf.funcs], axis=-1)
    comp.transfer_function.data = np.clip(rgba * 255, 0, 255).astype("u1").reshape(256, 1, 4)

    cam = scene.camera
    cam.update(position=[1.6, 1.1, 2.0], focus=[0.5, 0.5, 0.5], up=[0, 1, 0], fov=45.0, aspect_ratio=1.0)

    baseline = rc.run()
    yt.write_bitmap(baseline[:, :, :3], "19_final_baseline.png")

    position = np.array(cam.position, dtype=np.float64)
    focus = np.array(cam.focus, dtype=np.float64)
    up = np.array(cam.up, dtype=np.float64)
    forward = focus - position
    forward /= np.linalg.norm(forward)
    right = np.cross(forward, up)
    right /= np.linalg.norm(right)
    true_up = np.cross(right, forward)

    fov_rad = np.radians(cam.fov)
    half_h = np.tan(fov_rad / 2.0)
    half_w = half_h * cam.aspect_ratio

    xs = (np.arange(W) + 0.5) / W * 2.0 - 1.0
    ys = 1.0 - (np.arange(H) + 0.5) / H * 2.0  # row 0 = top -> ys[0] = +1 (up)
    xx, yy = np.meshgrid(xs, ys)

    directions = (
        forward[None, None, :]
        + xx[..., None] * half_w * right[None, None, :]
        + yy[..., None] * half_h * true_up[None, None, :]
    )
    directions /= np.linalg.norm(directions, axis=-1, keepdims=True)

    dist_to_focus = np.linalg.norm(focus - position)
    # Small, off-center sphere so it only partly overlaps the blob.
    sphere_center = position + forward * dist_to_focus * 0.6 + right * 0.13
    sphere_radius = 0.13

    oc = position[None, None, :] - sphere_center[None, None, :]
    b = 2.0 * np.einsum("hwc,hwc->hw", directions, oc)
    c = np.einsum("hwc,hwc->hw", oc, oc) - sphere_radius**2
    a = np.einsum("hwc,hwc->hw", directions, directions)
    disc = b**2 - 4 * a * c
    hit = disc >= 0
    sqrt_disc = np.sqrt(np.maximum(disc, 0))
    t_candidate = (-b - sqrt_disc) / (2 * a)
    valid = hit & (t_candidate > 0)

    t_hit = np.where(valid, t_candidate, 1e6)
    hit_points = position[None, None, :] + t_hit[..., None] * directions
    proj = np.array(cam.projection_matrix)
    view = np.array(cam.view_matrix)
    hit_h = np.concatenate([hit_points, np.ones((H, W, 1))], axis=-1)
    clip = np.einsum("ij,hwj->hwi", proj @ view, hit_h)
    ndc_z = clip[..., 2] / clip[..., 3]
    window_depth = np.where(valid, 0.5 * ndc_z + 0.5, 1.0).astype("f4")

    import sys
    sys.path.insert(0, "/home/mturk/yt/cycles-yt/src")
    try:
        from cycles_yt import yt_idv_renderer as yir
        yir.set_external_depth_clip(scene, window_depth)
    except ImportError:
        # standalone fallback if the add-on source isn't on sys.path here
        comp.external_depth_texture = Texture2D(data=window_depth[::-1, :].T)
        comp.use_external_depth_clip = True

    clipped = rc.run()
    print("sphere clip differs from baseline:", not np.allclose(baseline, clipped))
    yt.write_bitmap(clipped[:, :, :3], "20_final_sphere_clipped.png")

    comp.use_external_depth_clip = False
    restored = rc.run()
    print("disabling clip restores baseline:", np.allclose(baseline, restored))


verify_axis_convention()
uniform_wall_test()
sphere_shaped_clip_test()
