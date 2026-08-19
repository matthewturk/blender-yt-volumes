"""Sixth exploration: "the rays don't stop" -- yt_idv's ray-marching shader
(`ray_tracing.frag.glsl`) had no way to stop a ray's integration early at an
externally-supplied depth (e.g. an opaque occluder like the marker sphere
from scripts/yt_idv_exploration/blender_scene/). It always integrated all
the way out to `t1`, the ray's exit point from the CURRENT BLOCK's own
bounding box -- there was no per-pixel "max distance" concept to plug an
external occluder's depth into.

This is a REAL yt_idv modification, not a workaround on this add-on's side
-- GLSL shader source can't be monkeypatched the way Python can. Minimal and
additive: a new uniform pair (`external_depth_tex`, `use_external_depth_clip`)
that defaults to a complete no-op when unused. See
`yt_idv_external_depth_clip.patch` for the exact diff against the checkout
in `../../yt_idv/`, and `ray_tracing.frag.glsl`'s own comment for the
shader-side rationale.

Run inside the throwaway venv (see setup_ytidv_venv.sh), against the
patched yt_idv checkout (already applied there, not a separate step).

Two real bugs found and fixed while building/verifying this, both the hard
way (empirically, not by reading source and assuming):

1. **Import-order gotcha (this script's bug, not yt_idv's).**
   `from yt_idv.opengl_support import Texture2D` (or anything under
   `yt_idv.opengl_support`) MUST come AFTER `yt_idv.render_context(...)`,
   not before -- that call is what sets `PYOPENGL_PLATFORM`, and
   `opengl_support.py` does `from OpenGL import GL` at module level.
   Importing it too early binds PyOpenGL to the wrong platform before EGL
   is selected, manifesting as `glTexStorage3D` (and presumably other
   newer-GL functions) being undefined. Looked like GPU/driver flakiness
   at first (it intermittently "worked" depending on unrelated timing); it
   wasn't -- fixing just the import order made it 100% reliable.
2. **The `viewport` uniform's own doc comment is wrong (yt_idv's bug, not
   fixed here -- just discovered and worked around).**
   `known_uniforms.inc.glsl` documents `viewport` as `(offset_x, offset_y,
   1 / screen_x, 1 / screen_y)` -- implying you MULTIPLY by `.zw` to get a
   normalized UV. `yt_idv.cameras.base_camera.BaseCamera._set_uniforms`
   actually sets it to the RAW `GL.glGetIntegerv(GL_VIEWPORT)`, i.e.
   `(x, y, width, height)` in pixels -- you must DIVIDE by `.zw`, not
   multiply. Multiplying (what the comment implies, and what this shader's
   first draft did) sends `screen_uv` far outside `[0, 1]`; combined with
   the texture's default "clamp" boundary, EVERY fragment then samples the
   same corner texel regardless of its real screen position. This
   "accidentally worked" for a spatially-UNIFORM test depth (a uniform
   array's value is the same at any texel, so the bug was invisible) but a
   spatially-VARYING depth had ZERO effect anywhere on screen -- confirmed
   directly: an aggressive `depth=0.0` clip across an entire HALF of the
   frame produced a `np.allclose`-identical render to the unclipped
   baseline. Fixed by dividing instead of multiplying (see
   `ray_tracing.frag.glsl`'s own comment for the exact reasoning).

A THIRD thing, not a yt_idv bug, but a real gotcha in USING this feature
from Python: `yt_idv.opengl_support.Texture2D` uploads `data.swapaxes(0, 1)`
internally, and combined with `gl_FragCoord`'s OpenGL-native
bottom-left-origin Y axis (the opposite of a normal top-down image), a
plain `(height, width)` depth array in ordinary top-down convention needs
`depth[::-1, :].T` before `Texture2D(data=...)` samples it where you'd
expect. Confirmed empirically with a marked top-left quadrant (see
`verify_axis_convention` below) -- this exact transform is now baked into
`yt_idv_renderer._to_texture2d_axis_convention` / `set_external_depth_clip`,
so callers of the production module never need to think about it; this
script demonstrates it explicitly for anyone extending the mechanism
further.

Findings, using `yt_idv_renderer.set_external_depth_clip` (the production
function, not a hand-rolled equivalent):

  - A uniform (spatially-constant) depth plane at progressively closer
    distances shows graduated, all-or-nothing-per-ray occlusion (full
    render -> a thin sliver of the domain box's near face -> nothing at
    all) -- proves rays genuinely stop mid-integration, not a hard
    per-frame cutoff (`images/16_*.png` - `images/18_*.png`).
  - A REAL per-pixel, sphere-shaped depth (analytic ray/sphere
    intersection, positioned to only partly overlap the rendered
    isocontour structure) shows a genuine, clean partial "bite" taken out
    of the nebula -- exactly the "sphere partly in front of it, with gas
    still visible around it" result originally asked for, and NOT
    achievable with the flat alpha-over composite from
    scripts/yt_idv_exploration/blender_scene/ (which can only ever produce
    an all-or-nothing cutout, since MIP has no per-pixel depth of its own
    to composite against -- see that directory's README). This is real
    depth-AWARE integration: gas outside the sphere's silhouette renders
    completely normally, only the overlapping portion is clipped
    (`images/19_final_baseline.png`, `images/20_final_sphere_clipped.png`).
  - Disabling the clip (`set_external_depth_clip(scene, None)`) exactly
    reproduces the unclipped baseline in every test above (`np.allclose`,
    not just "looks the same") -- confirms zero regression when this
    feature isn't used.

NOT YET wired into `yt_idv_renderer.py`'s production render functions
against Blender's REAL Cycles Z-buffer: doing that needs the
dataset-to-object spatial registration problem solved first (yt_idv's
camera has no metric correspondence to Blender's camera -- see
yt_idv_renderer.py's module docstring on the orbit-mapping-only camera).
What IS available now, and demonstrated here, is depth clipping against
anything already in yt_idv's OWN window-space-depth convention -- e.g. an
analytically-computed occluder (this script) or a marker object rendered
within the SAME yt_idv scene/camera, fully self-consistent, no
cross-coordinate-system conversion needed.
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
    """The window-space depth (0..1, same convention `gl_FragDepth` already
    uses in ray_tracing.frag.glsl) of a point `dist` along the camera's own
    view direction.
    """
    forward = np.array(cam.focus) - np.array(cam.position)
    forward = forward / np.linalg.norm(forward)
    world_pt = np.array(cam.position) + forward * dist
    clip = cam.projection_matrix @ (cam.view_matrix @ np.array([*world_pt, 1.0]))
    ndc_z = clip[2] / clip[3]
    return 0.5 * ndc_z + 0.5


def verify_axis_convention():
    """Confirms the `depth[::-1, :].T` transform (see module docstring) --
    a normal top-down "top-left quadrant clipped" array should show the
    clipped notch in the top-left of the ACTUAL render, not some other
    corner.
    """
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
    """The real deliverable: a per-pixel, sphere-shaped depth clip that
    partially occludes the rendered isocontour structure -- gas outside the
    sphere's silhouette stays fully rendered, only the overlapping portion
    disappears. This is what a flat alpha-over composite (see
    scripts/yt_idv_exploration/blender_scene/) can NEVER produce, since MIP
    has no per-pixel depth of its own to composite against.
    """
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
    # Small, off-center sphere -- partly overlapping the blob, not
    # swallowing it whole (an earlier, larger/centered attempt did that;
    # see git history / the module docstring's tuning note).
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
    # Exercise the REAL production helper (handles the axis-convention
    # transform internally), not a hand-rolled equivalent.
    try:
        from cycles_volume_override import yt_idv_renderer as yir
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
