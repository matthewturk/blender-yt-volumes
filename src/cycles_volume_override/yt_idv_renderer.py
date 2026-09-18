"""Stub external renderer that calls the REAL yt_idv volume renderer
(offscreen OpenGL) instead of the toy/debug renderers in
`sample_renderers.py`/`external_render.py`. Two render modes:
`yt_idv_max_intensity_renderer` (Maximum Intensity Projection) and
`yt_idv_transfer_function_renderer` (a real transfer function -- isocontour
layers at fixed densities in log space, the standard first cut per the
brief this was extended against). Both share `_render_with_yt_idv` for
everything except how the `block_rendering` component itself is configured.

Deliberately scoped down, per the brief this was written against:

  - Always renders one hardcoded sample dataset (`yt.load_sample(
    "IsolatedGalaxy")`), NEVER `request.volume.dataset.dataset_path`/
    `dataset_kind`. Wiring up real dataset selection/loading is future work,
    explicitly out of scope here.
  - **Now DOES depth-clip against `request.depth` -- the flagged object's
    own `bounding_box_local`/`matrix_world` (already part of
    `VolumeObjectParams`, no new field needed) is the registration between
    Blender world space and yt_idv's `[0, 1]^3` "unitary" domain space**: a
    unitary point `u` lives at object-local point `bbox_min + u*bbox_extent`,
    then `matrix_world` into Blender world space. This single registration
    is used for BOTH the camera (position/orientation now genuinely
    transformed through it, not just orientation) and the depth clip
    (Cycles' Z-buffer reconstructed into world-space hit points, transformed
    through the SAME registration into unitary space, then projected through
    yt_idv's own now-correctly-positioned camera to get its window-space
    depth convention) -- see `_registered_camera_kwargs`/
    `_cycles_depth_to_yt_idv_window_depth`. This is exactly the "just use a
    transform to unitary space" approach -- no new yt_idv Camera subclass
    needed, `BaseCamera.update(position=..., focus=..., up=..., ...)` was
    already general enough, it just needed correctly-transformed values fed
    into it instead of an orbit heuristic.

    Still an APPROXIMATION, and still says so honestly: (1) yt_idv renders
    SQUARE (aspect 1.0, see `_render_with_yt_idv`'s own docstring on bug 3),
    while Blender's real camera is essentially never square -- the depth
    reconstruction happens on Blender's own (non-square) pixel grid using
    Blender's real projection, then gets center-padded onto the square
    canvas the same way the final image gets center-CROPPED back down, so
    the two alignments match, but a square-aspect yt_idv camera cannot
    reproduce Blender's true horizontal FOV, only its vertical one. (2) the
    object's bounding box has to actually correspond to where you want the
    dataset positioned -- it's a registration you place by sizing/moving
    the flagged object, not something inferred from the dataset itself.
  - Blender's camera FOV still uses the same `_approx_vertical_fov_deg`
    approximation as before (sensor/lens-derived, not a true frustum match)
    -- unrelated to the registration transform, a separate approximation.
  - The returned image's RGBA is yt_idv's own OpenGL framebuffer output,
    passed through close to verbatim (only row-flipped to match
    `request.depth.row_order`) and labeled `alpha_mode="PREMULTIPLIED"` --
    that label is a reasonable approximation, not a rigorously verified one:
    background pixels are genuine `(0, 0, 0, 0)` (confirmed correct end to
    end with a real Cycles render's own alpha -- see bug 4 below, the EGL
    context was missing an alpha channel entirely until fixed; valid under
    either premultiplied or straight-alpha reading either way), but yt_idv's
    `max_intensity` shader blends with `GL_MAX` (see shaders/shaderlist.yaml's
    `block_rendering` component), not the "over" operator this add-on's own
    Alpha Over compositing assumes, so the RGB/alpha relationship at
    partially-covered pixels hasn't been checked against what "premultiplied"
    strictly requires. Revisit once there's a real transfer function to
    reason about instead of a placeholder colormap.

Manually verified end-to-end -- real yt_idv, a real offscreen OpenGL
context, real IsolatedGalaxy sample data actually rendered and visually
inspected, not just imported -- in a throwaway Python 3.12 venv (`bpy` isn't
installable there -- see "NOT wired into this repo's own pytest suite"
below -- so this whole module's `yt_idv`-dependent path was exercised with a
hand-built `RenderRequest` standing in for what `operators.py` would build,
not through real Blender). `pip install -e <this checkout of yt_idv> pooch
pandas h5py`, then `yt_idv.render_context(backend, width=..., height=...)`,
`rc.add_scene(ds.all_data(), "density", no_ghost=True)` (confirmed
`component.render_method` already defaults to `"max_intensity"` --
`yt_idv/shaders/shaderlist.yaml`'s `component_shaders.block_rendering.
default_value`), camera repositioning via `camera.update(...)`, `rc.run()`.
Produced a real image: the domain box's silhouette plus a MIP-colormapped
blob at the galaxy's density peak. See `scripts/yt_idv_exploration/` for the
retained reproduction scripts and reference images (including the two real
yt_idv bugs below, caught only because that exploration deliberately used a
NON-square resolution -- every one of yt_idv's own examples/tests uses a
square one, which is exactly why these went unnoticed upstream).

**Real yt_idv bugs found while doing this, not this add-on's bugs to fix,
but load-bearing for the decisions below:**

1. `yt_idv.scene_graph.SceneGraph.image` reads the framebuffer via
   `GL.glReadPixels(0, 0, width, height, GL.GL_RGBA, GL.GL_FLOAT)`, which
   empirically returns shape `(width, height, 4)`, then calls
   `arr.swapaxes(0, 1)` **without reassigning the result** -- `ndarray.
   swapaxes` returns a new view, it doesn't mutate in place, so that line is
   a complete no-op and every non-square render comes back axis-swapped
   relative to the conventional `(height, width, 4)` image shape. Invisible
   in yt_idv's own examples/tests because they all use square dimensions
   (e.g. `width=height=1024`), where swapped-or-not is the same shape.
   Worked around entirely on this side -- no yt_idv edit needed -- by
   `_to_row_order` reshaping to `(height, width, 4)` before anything else
   touches the array.
2. The **OSMesa** backend (`yt_idv.rendering_contexts.osmesa_context.
   OSMesaRenderingContext`) produces genuinely CORRUPTED (not just
   mis-shaped) pixel data at non-square resolutions -- confirmed visually:
   distinct horizontal tiling/repetition of the whole scene, not a clean
   transpose of a correct image. Tried the obvious suspect (`OSMesaMakeCurrent`'s
   `width`/`height` arguments look swapped relative to the real OSMesa C API)
   and it did NOT fix it, so the actual cause is still open -- something
   deeper in how OSMesa's software rasterizer is being fed this buffer.
   **EGL does not have this problem** (its pixel data is clean at non-square
   sizes, only needs the axis-order fix above) -- that's why `DEFAULT_BACKEND`
   below is `"egl"`, not `"osmesa"`, despite EGL needing a working GPU +
   driver stack that OSMesa doesn't. Left uninvestigated further rather than
   chasing Mesa internals for what's still an exploratory stub; worth a real
   yt_idv fix (and an upstream bug report) before OSMesa is usable here.

3. A THIRD bug, found while testing camera motion (see
   `scripts/yt_idv_exploration/blender_scene/`): even through EGL (bug 1
   fixed), non-square renders show a persistent, EXACT period-4 banding
   pattern along every row -- confirmed via a raw scanline dump:
   `[value, 0, 0, value]` repeating, identical regardless of
   `component.sample_factor` up to 10x (ruling out a ray-marching
   under-sampling explanation -- tested and rejected, not just assumed).
   Confirmed absent at a SQUARE resolution with the same camera/dataset --
   only the aspect ratio changed between the clean and banded case. Root
   cause not identified (candidates: EGL surface/multisample config, a
   width/height mixup somewhere in yt_idv's own viewport or grid-expansion
   geometry shader specifically for non-square viewports). **Worked around
   here by rendering SQUARE and center-cropping** -- see
   `yt_idv_max_intensity_renderer`'s own docstring for the mechanics and
   cost (a FOV approximation gets slightly worse) -- rather than emitting a
   banded non-square image directly.

4. **EGL context missing an alpha channel entirely** -- found while
   verifying the real-Cycles sphere-and-volume demo's background alpha (see
   `scripts/yt_idv_exploration/blender_scene/render_real_demo.py`). `yt_idv/
   rendering_contexts/egl_context.py`'s `config_attribs` requested
   `EGL_RED_SIZE`/`EGL_GREEN_SIZE`/`EGL_BLUE_SIZE` = 8 each but never
   `EGL_ALPHA_SIZE` -- so the EGL pbuffer surface's default framebuffer had
   NO alpha bits, and every pixel `scene_graph.SceneGraph.image`'s
   `glReadPixels(..., GL_RGBA, ...)` fallback (used whenever `scene.fb is
   None`, i.e. every top-level offscreen render this add-on does) read back
   came back with alpha exactly `1.0`, genuinely-empty background pixels
   included -- confirmed empirically (`image[..., 3].min() == max() == 1.0`
   across an entire render, RGB still correctly `(0, 0, 0)` at background).
   This silently broke Alpha-Over compositing: a "transparent" volume layer
   composited as opaque black wherever yt_idv drew nothing. THIS repo's
   own tests never caught it (they don't check absolute alpha values against
   real GL output, only the `RenderResult.alpha_mode` contract label) --
   found only by actually eyeballing a real composited image and asking "is
   the background really transparent, not just labeled that way". Fixed by
   adding `EGL_ALPHA_SIZE, 8` to `config_attribs` -- see the patch file
   below; the RGB/red/green/blue/depth/stencil sizes were otherwise already
   correct. Root cause, not a workaround: unlike bugs 1-3 above, this one
   needed an actual yt_idv source edit (an EGL config request, not
   Python-side array logic this add-on could paper over).

See `scripts/yt_idv_exploration/README.md` for the fuller writeup and
reference images of all four, including the ones that led to ruling out
`sample_factor` for bug 3.

**External depth clip -- a real yt_idv shader modification, not a bug fix.**
Raised directly: a solid occluder placed "in front of" yt_idv content had no
way to actually stop the ray march early -- `ray_tracing.frag.glsl`'s loop
always integrates all the way to `t1` (the ray's exit from the CURRENT
BLOCK's own bounding box), with no concept of an externally-supplied max
depth. GLSL source can't be monkeypatched the way Python can, so this
required actually editing `yt_idv/shaders/ray_tracing.frag.glsl` (+
`known_uniforms.inc.glsl`) and `yt_idv/scene_components/blocks.py` -- see
`scripts/yt_idv_exploration/yt_idv_external_depth_clip.patch` for the exact
diff against the checkout in `../../yt_idv/`. New uniform pair
(`external_depth_tex`, `use_external_depth_clip`), off by default -- a
complete no-op unless explicitly enabled, confirmed by `use_external_
depth_clip = False` reproducing the unclipped baseline exactly
(`np.allclose`, not just "looks the same").

Found ANOTHER real yt_idv bug while wiring this up: `known_uniforms.inc.glsl`
documents the `viewport` uniform as `(offset_x, offset_y, 1 / screen_x,
1 / screen_y)` (implying you multiply by `.zw` for a normalized UV) -- but
`yt_idv.cameras.base_camera.BaseCamera._set_uniforms` actually sets it to
the RAW `GL.glGetIntegerv(GL_VIEWPORT)`, i.e. `(x, y, width, height)` in
pixels. The comment is wrong; you must DIVIDE by `.zw`, not multiply.
Multiplying (what the comment says, and what this shader's first draft
did) sends the computed screen UV far outside `[0, 1]`, and with the
texture's default "clamp" boundary, EVERY fragment ends up sampling the
same corner texel regardless of its real screen position. This
"accidentally worked" for a spatially-UNIFORM test depth (any single texel
equals a uniform array's value everywhere) but a spatially-VARYING depth
had ZERO effect anywhere -- confirmed directly: an aggressive `depth=0.0`
clip across an entire HALF of the frame produced a `np.allclose`-identical
render to the unclipped baseline, until this was found and fixed (divide,
not multiply -- see the shader's own comment).

A separate, non-bug gotcha in USING the feature from Python: `yt_idv.
opengl_support.Texture2D` uploads `data.swapaxes(0, 1)` internally, and
combined with `gl_FragCoord`'s OpenGL-native bottom-left-origin Y axis, a
plain `(height, width)` depth array in ordinary top-down image convention
needs `depth[::-1, :].T` before `Texture2D(data=...)` samples it where
you'd expect. Confirmed with a marked top-left quadrant landing top-left in
the actual render only after this transform -- now baked into
`_to_texture2d_axis_convention`/`set_external_depth_clip` below, so callers
never need to think about it.

Verified three ways, in order, in `scripts/yt_idv_exploration/
explore_external_depth_clip.py`:
  1. The axis-convention transform itself (a marked quadrant lands where
     expected).
  2. A spatially-uniform flat "wall" at progressively closer distances:
     full render -> a thin sliver of the domain box's near face (ray
     genuinely stopped mid-integration, not an all-or-nothing cutoff) ->
     nothing at all once the wall sits in front of the box entirely.
  3. **The real deliverable**: a per-pixel, sphere-shaped depth (analytic
     ray/sphere intersection, positioned to only partly overlap the
     rendered isocontour structure) produces a genuine, clean partial
     "bite" taken out of the nebula -- gas OUTSIDE the sphere's silhouette
     renders completely normally, only the overlapping portion is clipped.
     This is real depth-AWARE integration, and something the flat
     alpha-over composite in `scripts/yt_idv_exploration/blender_scene/`
     can never produce (that composite has no way to let gas in FRONT of
     an occluder stay visible while gas BEHIND it disappears -- MIP has no
     per-pixel depth of its own to composite against). See
     `images/19_final_baseline.png`/`images/20_final_sphere_clipped.png`.

**Two real bugs in THIS ADD-ON's own code, NOT yt_idv, found while building
the real end-to-end sphere-and-volume demo and BOTH FIXED.** Found via
`scripts/yt_idv_exploration/blender_scene/render_real_demo.py`, whose sphere
sits partly INSIDE the flagged box (unlike every previous depth-clip test
above, whose occluders sat entirely in front of the domain's near face): a
chunk of pixels genuinely occluded by the sphere (real Cycles beauty pass
confirmed real, correctly-Z'd sphere material there) still got FULLY OPAQUE
volume color drawn on top (`alpha == 1.0` exactly) -- identically in BOTH
`max_intensity` and `projection` render methods (only the leaked color
differed per method, which is why it read as an obvious light-pink patch in
projection mode but was easy to mistake for a dim material tone in MIP).
First suspected as a yt_idv shader/ray-march bug (an occluder embedded
inside the domain, rather than entirely in front of it, seemed like an
untested code path); investigated with a pure analytic ray/sphere depth
field, no Blender/Cycles involved at all (`scripts/yt_idv_exploration/
explore_bug6_embedded_occluder_leak.py`), which made it possible to inspect
the actual numbers going into the clip, not just the final image.

**The actual root cause, and the one that mattered: `_render_with_yt_idv`
called `scene.camera.update(**kwargs)` but never followed it with
`scene.camera._update_matrices()`.** `yt_idv.cameras.trackball_camera.
TrackballCamera` overrides `_compute_matrices` (the method `BaseCamera`'s
own trait-change observer calls whenever `position`/`focus`/`up`/`fov`/
`near_plane`/`far_plane`/`aspect_ratio` change) to a literal no-op (`pass`)
-- so calling `.update(...)` alone sets those TRAITS correctly but never
actually recomputes `projection_matrix`/`view_matrix` from them. yt_idv's
OWN internal code (`scene_graph._update_scene_camera_for_geometry`) always
follows `camera.update(...)` with an explicit `camera._update_matrices()`
for exactly this reason; this add-on's own camera-update call was missing
it. The effect was masked for months of this project's own development:
`view_matrix` "accidentally" still looked right, because its own
`@traitlets.default` computes lazily from whatever position/focus/up values
are CURRENT the first time something reads it, and nothing read it before
our own `.update()` call -- but `projection_matrix` has no such lazy
default, so it silently kept whatever it was computed as the very FIRST
time (effectively `add_scene`'s own construction-time default: `fov=45`,
`near_plane=0.001`, some initial `far_plane`), completely ignoring every
fov/near/far value this module ever computed and passed to `.update()`.
Confirmed directly: `scene.camera.projection_matrix` compared against
`get_perspective_matrix(fov, aspect_ratio, near_plane, far_plane)` (built
from the CURRENT, correctly-set trait values) mismatched before adding the
`_update_matrices()` call, matched exactly after. Every yt_idv render this
project has ever produced had the correct camera POSITION/ORIENTATION but
the WRONG (stale-default) field of view and near/far planes the entire
time, without this being visually obvious -- 45 degrees isn't wildly
different from the ~40-58 degree FOVs this module was actually computing,
so nothing looked obviously broken until a near/far-sensitive feature (the
external depth clip) exposed it. **Fixed** by adding the missing
`scene.camera._update_matrices()` call right after `.update(...)` in
`_render_with_yt_idv`. Confirmed to resolve the real demo's leak completely
(`images/24_real_demo_sphere_and_volume.png` re-rendered clean, sphere
showing only its own real, unleaked shading) -- and also visibly corrects
this project's camera framing going forward (tighter, more accurate FOV
than every previous render in this exploration used).

**A second, independent, smaller issue found and fixed along the way:**
`_registered_camera_kwargs`'s near/far-plane heuristic (`near_plane = 0.01 *
look_ahead`, `far_plane = look_ahead * 4.0`, where `look_ahead` is the
distance from the camera to the DOMAIN CENTER) produced a near/far ratio of
400:1 regardless of how big the domain actually is relative to that
distance -- and since OpenGL's standard perspective projection maps
view-space depth to NDC depth nonlinearly (`~1/z`), an oversized near/far
range needlessly compresses the domain's usable depth range (for this
demo's exact camera/geometry: the domain's nearest/farthest corners, at
unitary distance 2.08/3.79 from the camera, mapped to window depths of only
0.9884/0.9948 -- the WHOLE domain's usable range squeezed into 0.006 out of
the full `[0, 1]`). This alone was NOT suffient to explain the observed
leak (confirmed: fixing this formula without ALSO adding the missing
`_update_matrices()` call had no effect, since the corrected near/far never
reached `projection_matrix` either way) -- but it's still a real, worthwhile
precision improvement on its own, independent of the bug above. **Fixed**
in `_near_far_planes`: near/far are now sized around the ACTUAL domain (the
unitary `[0, 1]^3` cube's own 8 corners), not an arbitrary multiple of
distance-to-center -- `near_plane` just inside the nearest corner,
`far_plane` just past the farthest, both with a small fixed margin.

`explore_bug6_embedded_occluder_leak.py` and its two reference images
(`images/26_*`/`images/27_*`) are kept as-is, describing the ORIGINAL,
now-fixed investigation for the record -- see that script's own updated
docstring for why re-running it doesn't cleanly reproduce the original
symptom anymore (the corrected camera framing changes what that specific
synthetic sphere position even means relative to the dataset, so it's no
longer a reliable repro; the REAL, user-facing case -- `render_real_demo.py`
-- is the one that was actually confirmed fixed).

Exposed at this module's level as `set_external_depth_clip(scene,
depth_or_none)`. **Now wired into both renderer functions against Blender's
real Cycles Z-buffer**, via the object-bounding-box registration described
in the module docstring's scope section above
(`_cycles_depth_to_yt_idv_window_depth`) -- the dataset-to-object spatial
registration problem this used to be blocked on is the SAME registration
now used for camera positioning too, so once that existed for the camera,
depth-clipping Cycles' own Z-buffer through it was the natural next step,
not a separate unsolved problem. See `_render_with_yt_idv`'s docstring for
exactly how the (possibly non-square) Blender depth grid gets aligned with
yt_idv's forced-square render canvas.

NOT wired into this repo's own default pytest run beyond the pure-Python
helpers below (`_camera_update_kwargs`, `_approx_vertical_fov_deg`,
`_to_row_order`) -- but CAN be, in the same `bpy`/Python-3.13 dev venv this
repo already uses for everything else, via the optional `ytidv` dependency
group (`uv sync --group ytidv`; see pyproject.toml and README "yt_idv
integration"). This used to be impossible: PyPI's yt_idv release (0.5.4)
lists `imgui` as an unconditional install dependency, which has no cp313
wheel and fails to build from source against 3.13's changed C API -- and
this repo's dev venv is pinned to Python 3.13 to match the pip `bpy`
wheel's ABI, so there was no single Python version that could hold both.
Fixed upstream 2026-09-15 (yt-project/yt_idv#268, "imgui-optional"): imgui
moved to an opt-in `gui` extra (only needed for the interactive
pyglet+imgui control panel -- never for this module's own `egl`-backend,
no-GUI path), so a plain install no longer needs it. Confirmed empirically:
`bpy` and real `yt_idv` now import together in ONE Python 3.13 interpreter,
a real offscreen EGL context renders through it, and
`scripts/yt_idv_exploration/blender_scene/render_real_demo.py` runs the
add-on's actual two-pass pipeline against real Cycles + real yt_idv content
end to end this way, with no subprocess bridge -- see that script's module
docstring. Not yet in a PyPI release as of this writing (0.5.4 predates it),
hence the git dependency in the `ytidv` group.

**A real, separate incompatibility found while making that in-process run
work, NOT a yt_idv bug:** Blender's own realtime compositor (the default
`scene.render.compositor_device == "GPU"`) grabs a GPU/EGL context as soon
as ANY render runs through a scene's compositor node tree -- confirmed via
a minimal repro isolating exactly this (a bare Cycles render: fine; the
same render through nothing more than a Render Layers -> File Output graph,
i.e. exactly this add-on's own pass 1: fails). Once that's happened in this
process, yt_idv's own subsequent `yt_idv.render_context("egl", ...)` fails
with `EGLError(err=EGL_BAD_ACCESS, baseOperation=eglMakeCurrent, ...)` --
apparently Blender's own EGL context stays bound in a way `eglMakeCurrent`
for a second, unrelated context can't share. `operators.render_current_frame`
now forces `scene.render.compositor_device = "CPU"` for the duration of a
yt_idv-mode render specifically (restored afterward) to avoid this -- see
its own docstring/comment for the confirmation. `yt`/`yt_idv` are still
imported lazily here (inside functions, not at module level) so importing
this module -- and therefore registering the whole add-on -- never requires
them installed; only actually selecting a yt_idv renderer mode and
rendering does. See README "yt_idv integration" for installing yt_idv into
your real interactive Blender's own Python too (a separate install from
this repo's dev venv either way, but no longer blocked by the same
imgui/Python-version conflict).

Known rough edges, beyond the scope points above:
  - The dataset + offscreen GL context are cached at module level
    (`_get_context`) and never explicitly torn down -- there's no hook into
    `operators.unregister()` for this yet, so disabling/reloading the
    add-on leaks the GL context until Blender itself exits. Fine for
    exploring; worth a real fix before this is anything more than that.
  - `DEFAULT_BACKEND` is a hardcoded module constant, not an add-on
    preference. Edit it directly for now if EGL isn't viable on your
    machine (no GPU / no working driver stack) -- but see bug 2 above
    before reaching for `"osmesa"` as the fallback; it doesn't currently
    work for any resolution that isn't square.
  - `TF_VALUE_MIN`/`TF_VALUE_MAX`/`TF_N_LAYERS`/`TF_COLORMAP` are hardcoded
    constants tuned by trial and error for IsolatedGalaxy's density field
    (see "Transfer function" above) -- not derived from the field's own
    statistics, so they won't transfer to a different dataset/field without
    re-tuning.

Idea floated for later, not yet started: designing the transfer function
visually via a Blender Material node graph (e.g. a ColorRamp) instead of
`yt.ColorTransferFunction.add_layers()` parameters -- bake the material's
output to a small (256x1 or 2D) image, read it back, and feed it in as
`TransferFunctionTexture.data` (which is exactly `(nbins, 1, 4)` uint8 RGBA
already -- see `_build_isocontour_transfer_function_data`). Plausible
mechanically (Cycles bake -> `bpy.data.images` -> `.pixels` is a normal
Blender workflow), not attempted here.
"""

import math

import numpy as np

from .external_render import RenderRequest, RenderResult
from .sample_renderers import _persp_ortho_rays

SAMPLE_DATASET_NAME = "IsolatedGalaxy"
SAMPLE_FIELD = "density"
DEFAULT_BACKEND = "egl"  # NOT "osmesa" -- see module docstring's yt_idv bug #2

# --- Transfer function (isocontours at fixed densities, in log space) -----
#
# yt_idv's block_rendering component normalizes field data to [0, 1] by
# *linear* min-max across the WHOLE loaded dataset before it ever reaches
# the GPU (yt_idv.scene_data.base_data.BaseData._normalize_by_min_max) --
# `tf_min`/`tf_max`/the transfer function's own domain are in THIS
# normalized-fraction space, NOT the field's raw physical units (density in
# g/cm**3, spanning ~1e-31 to ~1e-23 for IsolatedGalaxy). Confirmed
# empirically: `component.data.min_val`/`.max_val` print the raw physical
# bounds, but the *useful* signal after normalization for this dataset/field
# sits around 1e-4 to 1e-2 of that [0, 1] range (cross-checked against
# max_intensity mode's own auto-computed `cmap_min`/`cmap_max`, which is the
# actual per-view max of this SAME normalized quantity) -- values near the
# top of the nominal [0, 1] range are essentially always empty for this
# dataset. TF_VALUE_MIN/MAX below bracket that observed range, found by
# trying (1e-4, 1.0) first (nearly all black -- see scripts/
# yt_idv_exploration/yt_idv_transfer_function/README.md) and narrowing until
# real isocontour structure appeared.
#
# `tf_log=True` (set below) makes the fragment shader take the natural log
# (GLSL `log()`, base e, NOT base 10) of both the sample and `tf_min`/
# `tf_max` before mapping into the transfer function texture -- so
# `yt.ColorTransferFunction`'s own `x_bounds` (and everything plotted against
# them, like `add_layers`' isocontour placements) must be `math.log(...)` of
# these same TF_VALUE_* bounds, not `log10`.
TF_VALUE_MIN = 1e-6
TF_VALUE_MAX = 2e-3
TF_N_LAYERS = 5  # how many isocontour layers add_layers() spaces evenly (in log space) between TF_VALUE_MIN/MAX
TF_COLORMAP = "viridis"
TF_NBINS = 256  # matches TransferFunctionTexture's expected (256, 1, 4) shape

_dataset_cache = {}  # dataset name -> (ds, data_source)
_context_cache = {}  # (backend, width, height) -> (render_context, scene)
_transfer_function_cache = {}  # (n_layers, colormap, value_min, value_max) -> (256, 1, 4) uint8 RGBA texture data


def _load_sample_source():
    """Lazily load the hardcoded sample dataset ONCE per Blender session.

    This is the expensive part (disk I/O + hierarchy parsing) -- the
    two-pass pipeline calls the render function once per flagged object per
    frame, so caching here isn't a nice-to-have, it's required for this to
    be usable at all across more than a single call.
    """
    cached = _dataset_cache.get(SAMPLE_DATASET_NAME)
    if cached is not None:
        return cached
    try:
        import yt
    except ImportError as exc:
        raise ImportError(
            "yt_idv rendering needs `yt` importable from Blender's own "
            "Python -- see README 'yt_idv integration' for how to install "
            "it there."
        ) from exc
    ds = yt.load_sample(SAMPLE_DATASET_NAME)
    data_source = ds.all_data()
    _dataset_cache[SAMPLE_DATASET_NAME] = (ds, data_source)
    return ds, data_source


def _get_context(backend, width, height):
    """Lazily create ONE offscreen render context + scene per (backend,
    width, height), reused across calls. yt_idv's offscreen contexts fix
    their framebuffer size at creation, so a resolution change (e.g. the
    user resizes the render) transparently gets its own cached context
    rather than reusing a wrongly-sized one.

    Does NOT set `component.render_method` -- since the SAME cached context
    can be reused by either renderer mode below (if the user switches the
    `renderer_mode` preference mid-session), each render call configures the
    component itself, every time, rather than relying on whatever was set
    when the context was first created.
    """
    key = (backend, width, height)
    cached = _context_cache.get(key)
    if cached is not None:
        return cached
    try:
        import yt_idv
    except ImportError as exc:
        raise ImportError(
            "yt_idv rendering needs `yt_idv` importable from Blender's own "
            "Python -- see README 'yt_idv integration' for how to install "
            "it there."
        ) from exc
    ds, data_source = _load_sample_source()
    rc = yt_idv.render_context(backend, width=width, height=height)
    scene = rc.add_scene(data_source, SAMPLE_FIELD, no_ghost=True)
    _context_cache[key] = (rc, scene, ds)
    return rc, scene, ds


def _to_texture2d_axis_convention(depth):
    """`Texture2D._set_data` uploads `data.swapaxes(0, 1)` (see
    `yt_idv.opengl_support.Texture2D`) while declaring the texture's GL
    dimensions from the PRE-swap shape -- combined with `screen_uv` in
    `ray_tracing.frag.glsl` being built from `gl_FragCoord` (OpenGL window
    coordinates: y=0 at the BOTTOM, unlike a normal top-down image) and
    yt_idv's OWN `SceneGraph.image` flipping ITS OWN OpenGL readback back to
    top-down afterward, the net effect is that a plain `(height, width)`
    array in ordinary top-down image convention (row 0 = top, matching e.g.
    `DepthBuffer.data` once row-order-normalized) needs this exact transform
    before `Texture2D(data=...)` samples it at the position you'd expect.
    Confirmed empirically, not derived purely from reading the source --
    see scripts/yt_idv_exploration/explore_external_depth_clip.py's
    "axis convention" test (a marked top-left quadrant in, confirmed to
    land top-left in the actual render only after this exact transform).
    """
    return depth[::-1, :].T


def set_external_depth_clip(scene, depth=None):
    """Enable/disable the external depth clip on every component in `scene`
    -- see module docstring's "External depth clip" section for what this
    is; wired into both renderer functions against Blender's real Z-buffer
    (`_cycles_depth_to_yt_idv_window_depth` below).

    `depth`: a `(height, width)` float32 array in ordinary top-down image
    convention (row 0 = top) and yt_idv's own window-space depth convention
    (0..1, same as what `gl_FragDepth` already uses in
    `ray_tracing.frag.glsl` -- NOT Blender/Cycles' depth convention, see
    `operators.DEPTH_CONVENTION`/`DEPTH_ROW_ORDER`, which is a different
    space entirely), or `None` to disable clipping and restore the
    unclipped behavior (confirmed to reproduce the exact unclipped
    baseline -- see scripts/yt_idv_exploration/explore_external_depth_clip.py).
    Must match the render's actual (square, post-`_render_with_yt_idv`)
    pixel dimensions.

    Raises if `depth` is given but NO component in `scene` actually
    supports the clip -- a real failure mode, not a hypothetical one: this
    feature needs yt_idv changes (yt-project/yt_idv#247, "Fixing for
    external usage") that are NOT on PyPI or `main` as of this writing (see
    README "yt_idv integration" and pyproject.toml's `ytidv` group for how
    to get a checkout that has them). Before this check existed, installing
    a plain/unpatched yt_idv made this function silently do nothing on
    every call (the `hasattr` guard below just skipped every component) --
    every render still "worked" and looked plausible, but the clip measured
    a flat 0.0% effect in EVERY renderer mode, not just MIP, until
    `scripts/yt_idv_exploration/blender_scene/verify_overlap.py` caught it
    by actually diffing a clipped vs. unclipped render instead of trusting
    that a plausible-looking image meant the clip was doing anything.
    """
    try:
        from yt_idv.opengl_support import Texture2D
    except ImportError as exc:
        raise ImportError(
            "yt_idv rendering needs `yt_idv` importable from Blender's own "
            "Python -- see README 'yt_idv integration' for how to install "
            "it there."
        ) from exc
    any_component_supports_clip = False
    for component in scene.components:
        if not hasattr(component, "use_external_depth_clip"):
            continue  # e.g. annotation components that aren't block_rendering
        any_component_supports_clip = True
        if depth is None:
            component.use_external_depth_clip = False
            continue
        texture_data = _to_texture2d_axis_convention(np.asarray(depth, dtype="f4"))
        component.external_depth_texture = Texture2D(data=np.ascontiguousarray(texture_data))
        component.use_external_depth_clip = True
    if depth is not None and not any_component_supports_clip:
        raise RuntimeError(
            "set_external_depth_clip(depth=...) was asked to enable the "
            "depth clip, but no component in this yt_idv scene has a "
            "`use_external_depth_clip` attribute -- this yt_idv install is "
            "missing the external-depth-clip changes (yt-project/"
            "yt_idv#247), so the clip would silently do nothing while every "
            "render still looked plausible. See this function's own "
            "docstring and README 'yt_idv integration' for how to get a "
            "yt_idv checkout that actually supports it."
        )


def _build_isocontour_transfer_function_data(value_min, value_max, n_layers, colormap, nbins=TF_NBINS):
    """`n_layers` isocontours at evenly-log-spaced densities between
    `value_min` and `value_max` (both in yt_idv's own [0, 1]-normalized
    value space -- see the module docstring's "Transfer function" section,
    NOT raw physical units), colored by `colormap` -- exactly
    `yt.ColorTransferFunction.add_layers`, the standard yt volume-rendering
    technique for "isocontours at fixed densities in log space" requested
    here. Returns the `(nbins, 1, 4)` uint8 RGBA array
    `yt_idv.opengl_support.TransferFunctionTexture.data` expects.
    """
    import yt

    tf = yt.ColorTransferFunction((math.log(value_min), math.log(value_max)), nbins=nbins)
    tf.add_layers(n_layers, colormap=colormap)
    rgba = np.stack([channel.y for channel in tf.funcs], axis=-1)  # (nbins, 4), [0, 1]
    return np.clip(rgba * 255.0, 0, 255).astype("u1").reshape(nbins, 1, 4)


def _get_isocontour_transfer_function_data():
    """Cached -- building the transfer function itself is cheap (milliseconds),
    but there's no reason to redo it every single frame when the settings
    (`TF_VALUE_MIN`/`MAX`/`TF_N_LAYERS`/`TF_COLORMAP`) haven't changed.
    """
    key = (TF_N_LAYERS, TF_COLORMAP, TF_VALUE_MIN, TF_VALUE_MAX)
    cached = _transfer_function_cache.get(key)
    if cached is not None:
        return cached
    data = _build_isocontour_transfer_function_data(TF_VALUE_MIN, TF_VALUE_MAX, TF_N_LAYERS, TF_COLORMAP)
    _transfer_function_cache[key] = data
    return data


def _approx_vertical_fov_deg(camera, aspect_ratio):
    """Approximate vertical FOV (full angle, degrees) for yt_idv's
    perspective-only camera (`yt_idv.cameras.trackball_camera.TrackballCamera`
    -- its `proj_func` is always `yt.utilities.math_utils.
    get_perspective_matrix`, no orthographic option exists there) from
    Blender's PERSP `CameraParams`.

    Simplified relative to Blender's own `sensor_fit` resolution (which also
    folds in `pixel_aspect_x`/`pixel_aspect_y`) -- fine here, where
    positional/scale correspondence is already approximate at best (see
    module docstring); not meant to be frame-accurate the way
    `operators.build_camera_params`'s `projection_matrix` is for Cycles
    itself.
    """
    if camera.sensor_fit == "VERTICAL":
        vert_fov = 2.0 * math.atan((camera.sensor_height_mm / 2.0) / camera.lens_mm)
        return math.degrees(vert_fov)
    horiz_fov = 2.0 * math.atan((camera.sensor_width_mm / 2.0) / camera.lens_mm)
    return math.degrees(2.0 * math.atan(math.tan(horiz_fov / 2.0) / aspect_ratio))


def _domain_array_unitary(value):
    """`ds.domain_center`/`ds.domain_width` -> a plain (3,) float32 array in
    yt_idv's own "unitary" convention (see
    `yt_idv.traitlets_support.YTPositionTrait`) -- a real yt `Dataset`
    reports these as a `YTArray` in `code_length` units, NOT unitary, and
    the two only coincide numerically for a dataset whose domain happens to
    be exactly 1 code_length wide (true of IsolatedGalaxy, NOT true in
    general) -- so this converts explicitly rather than assuming they match,
    mirroring what `yt_idv.cameras.trackball_camera.TrackballCamera`'s own
    `_get_camera_for_ds_geometry` relies on `YTPositionTrait` to do
    implicitly. Also accepts a plain array/tuple (no `.in_units`) unchanged,
    so the trivial stand-ins the unit tests use don't need to fake a real
    `YTArray`.
    """
    if hasattr(value, "in_units"):
        value = value.in_units("unitary").d
    return np.asarray(value, dtype="f4")


def _bbox_local_min_extent(bounding_box_local):
    """(min, extent) of `VolumeObjectParams.bounding_box_local` (the
    flagged object's own 8-corner bounding box, in OBJECT-local space) --
    the registration this module uses for BOTH the camera and the depth
    clip: a yt_idv "unitary" point `u` in `[0, 1]^3` lives at object-local
    point `min + u*extent`, then `matrix_world` into Blender world space.
    Degenerate (zero-extent) axes are guarded to 1.0 rather than dividing
    by zero -- a flat/degenerate flagged object just means that axis's
    registration is meaningless, not a crash.
    """
    corners = np.asarray(bounding_box_local, dtype=np.float64)
    local_min = corners.min(axis=0)
    local_max = corners.max(axis=0)
    extent = local_max - local_min
    return local_min, np.where(extent == 0.0, 1.0, extent)


def _registered_camera_kwargs(camera, volume, ds, width, height):
    """Real positional + orientation mapping from Blender's `CameraParams`
    to the kwargs for yt_idv's `BaseCamera.update(**kwargs)`, using
    `volume`'s own `bounding_box_local`/`matrix_world` as the registration
    between Blender world space and yt_idv's unitary domain space -- see
    module docstring's scope section and `_bbox_local_min_extent`. Replaces
    the earlier "orbit at a fixed heuristic distance, orientation only"
    approach entirely: position now comes from where the Blender camera
    ACTUALLY sits relative to the flagged object, not an assumption.

    Pure Python/numpy, no `yt`/`yt_idv`/`bpy` import needed -- `ds` only
    needs `.domain_center` (a real yt `YTArray`, or a plain array/tuple
    stand-in for tests; see `_domain_array_unitary`), used only as a
    reference point for a sensible near/far/look-ahead distance, not for
    positioning itself anymore.
    """
    if camera.projection != "PERSP":
        raise NotImplementedError(
            "yt_idv's camera is perspective-only (yt_idv.cameras."
            "trackball_camera.TrackballCamera has no ortho/panoramic "
            f"projection) -- got CameraParams.projection={camera.projection!r}."
        )

    cam_matrix_world = np.array(camera.matrix_world, dtype=np.float64)
    cam_pos_world = cam_matrix_world[:3, 3]
    # Blender camera-local forward is -Z, up is +Y (same convention already
    # relied on elsewhere in this add-on -- see sample_renderers.py's
    # _equirect_rays docstring).
    cam_forward_world = -cam_matrix_world[:3, 2]
    cam_up_world = cam_matrix_world[:3, 1]

    obj_matrix_world = np.array(volume.matrix_world, dtype=np.float64)
    obj_matrix_world_inv = np.linalg.inv(obj_matrix_world)
    local_min, local_extent = _bbox_local_min_extent(volume.bounding_box_local)

    cam_pos_local_h = obj_matrix_world_inv @ np.array([*cam_pos_world, 1.0])
    cam_pos_local = cam_pos_local_h[:3] / cam_pos_local_h[3]
    position = (cam_pos_local - local_min) / local_extent

    # Direction vectors transform through the same registration's LINEAR
    # part only (no translation) -- correct for repositioning a viewing
    # direction consistently through an affine map, not the
    # inverse-transpose a surface normal would need.
    forward_local = obj_matrix_world_inv[:3, :3] @ cam_forward_world
    forward_unitary = forward_local / local_extent
    forward_norm = np.linalg.norm(forward_unitary)
    if forward_norm == 0.0:
        raise ValueError(
            "camera forward direction vanished after the registration transform -- "
            "check VolumeObjectParams.matrix_world/bounding_box_local for degeneracy."
        )
    forward_unitary = forward_unitary / forward_norm

    up_local = obj_matrix_world_inv[:3, :3] @ cam_up_world
    up_unitary = up_local / local_extent
    # Non-uniform local_extent can skew the angle between forward/up (it's
    # not a rigid transform in general) -- re-orthogonalize against the
    # already-normalized forward direction rather than pass a skewed up
    # into get_lookat_matrix.
    up_unitary = up_unitary - np.dot(up_unitary, forward_unitary) * forward_unitary
    up_norm = np.linalg.norm(up_unitary)
    up_unitary = (up_unitary / up_norm) if up_norm > 0.0 else np.array([0.0, 1.0, 0.0])

    domain_center = _domain_array_unitary(ds.domain_center).astype(np.float64)
    look_ahead = max(float(np.linalg.norm(position - domain_center)), 1e-3)
    focus = position + forward_unitary * look_ahead

    aspect_ratio = width / height
    fov_deg = _approx_vertical_fov_deg(camera, aspect_ratio)

    return {
        "position": position.astype("f4"),
        "focus": focus.astype("f4"),
        "up": up_unitary.astype("f4"),
        "fov": fov_deg,
        "aspect_ratio": aspect_ratio,
        **_near_far_planes(position),
    }


_UNIT_CUBE_CORNERS = np.array(
    [[x, y, z] for x in (0.0, 1.0) for y in (0.0, 1.0) for z in (0.0, 1.0)],
    dtype=np.float64,
)


def _near_far_planes(position, margin=0.1):
    """Real bug found while building the real end-to-end sphere-and-volume
    demo (`scripts/yt_idv_exploration/blender_scene/render_real_demo.py`):
    the PREVIOUS heuristic here (`near_plane = 0.01 * look_ahead`,
    `far_plane = look_ahead * 4.0`, where `look_ahead` is the distance from
    the camera to the DOMAIN CENTER) produced a near/far ratio of ~400:1
    regardless of the domain's own actual size -- and since OpenGL's
    standard perspective projection maps view-space depth to NDC depth
    NONLINEARLY (`~1/z`), a near/far range that vastly overshoots how far
    away the domain actually is compresses the domain's ENTIRE visible
    depth range into a tiny band just below NDC depth 1.0. Confirmed
    directly for the real demo's camera/geometry: the domain's own nearest
    and farthest corners (unitary distance 2.08 and 3.79 from the camera)
    mapped to window depths 0.9884 and 0.9948 respectively under the old
    heuristic -- the ENTIRE volume's usable depth range squeezed into 0.006
    out of the full [0, 1] window-depth range. `_cycles_depth_to_yt_idv_
    window_depth`'s external-depth-clip values for a real occluder embedded
    in that same tiny band came out numerically indistinguishable from a
    ray's own natural far-block-exit depth (both computed independently
    through float32 arithmetic), so `ray_tracing.frag.glsl`'s `current_depth
    >= external_max_depth` comparison flipped essentially at random from one
    screen pixel to the next -- explaining the light-colored leak observed
    INSIDE the sphere's own silhouette in `images/24_real_demo_sphere_and_
    volume.png` (as opposed to the crescent-shaped GAP around/below it,
    which was always correct): full alpha=1.0 "leaked" render on some
    pixels, correctly-clipped alpha=0.0 on adjacent ones, with no
    lighting/colormap/Blender-registration explanation -- confirmed by
    reproducing the exact same leak with a pure analytic ray/sphere depth
    field, no Blender or Cycles involved at all (see
    `scripts/yt_idv_exploration/explore_bug6_embedded_occluder_leak.py`,
    which predates this fix and still describes/reproduces the old
    behavior for the record).

    Fixed by sizing near/far around where the domain ACTUALLY is (the
    `[0, 1]^3` unitary cube's own 8 corners, since that's what unitary space
    means by construction -- see `_bbox_local_min_extent`), not an arbitrary
    multiple of distance-to-center: `near_plane` just inside the nearest
    corner, `far_plane` just past the farthest one, both with a small
    absolute `margin` (in unitary units, default 0.1 -- about 6% of the unit
    cube's own diagonal) so the domain's own boundary isn't near/far-clipped
    away. If the camera sits inside the domain, `near_plane` can legitimately
    want to be at or below 0, hence the `1e-5` floor (unchanged from before).
    This keeps the near/far ratio close to what the scene's own geometry
    actually needs, regardless of how far away the camera is positioned --
    the opposite of the old heuristic, whose ratio was constant (4.0/0.01 =
    400) no matter where the camera sat.
    """
    dists = np.linalg.norm(_UNIT_CUBE_CORNERS - position, axis=1)
    return {
        "near_plane": max(1e-5, float(dists.min()) - margin),
        "far_plane": float(dists.max()) + margin,
    }


def _cycles_depth_to_yt_idv_window_depth(request, yt_idv_camera):
    """Reconstruct Blender/Cycles' own `request.depth` as a per-pixel
    window-space depth in yt_idv's OWN convention, ready for
    `set_external_depth_clip` -- the actual "wire the Z-buffer through"
    step, using the SAME `request.volume` registration as
    `_registered_camera_kwargs` (they have to agree, or the clip would land
    somewhere yt_idv's camera doesn't actually see it).

    Steps, per pixel:
      1. Reconstruct the Blender WORLD-SPACE point the Z-buffer's `PLANAR`
         depth value corresponds to, using the exact ray-generation
         already verified in `sample_renderers._persp_ortho_rays` (camera
         intrinsics/extrinsics only, no bpy) plus the affine relationship
         between a ray's parameter `t` and its view-space Z (both are
         affine in world position, so solving for `t` is one division, no
         iteration -- see the inline derivation below).
      2. Transform that world point into unitary space via `volume`'s
         registration (object-local bounding box, then `matrix_world`
         inverse).
      3. Project the unitary point through yt_idv's OWN
         `projection_matrix`/`view_matrix` (`yt_idv_camera`, already
         positioned by `_registered_camera_kwargs` for this same frame) to
         get yt_idv's window-space depth convention (0..1, matching
         `gl_FragDepth` in `ray_tracing.frag.glsl`).

    Background/no-hit pixels (`request.depth.no_hit_value`, ~1e10) get
    window depth `1.0` (no clip) rather than running through the above --
    their reconstructed "hit point" is meaningless.

    Returns a `(height, width)` float32 array in ordinary top-down image
    convention (row 0 = top) -- i.e. already in the shape
    `set_external_depth_clip` expects; row-order conversion from
    `request.depth.row_order` is handled here, callers don't need to.
    """
    camera = request.camera
    depth_buffer = request.depth
    volume = request.volume

    origins, directions = _persp_ortho_rays(camera, depth_buffer.row_order, depth_buffer.data.shape)

    cam_matrix_world_inv = np.linalg.inv(np.array(camera.matrix_world, dtype=np.float64))
    cam_z_row = cam_matrix_world_inv[2, :3]
    cam_z_translation = cam_matrix_world_inv[2, 3]
    # view-space Z is affine in world position (matrix_world_inv has no
    # perspective row), and world position is affine in ray parameter t
    # (origin + t*direction) -- so view-space Z is affine in t too:
    # cam_z(t) = cam_z(origin) + t * cam_z_direction. PLANAR depth is
    # -cam_z (see sample_renderers._planar_depth), so solve
    # -depth == cam_z(origin) + t*cam_z_direction for t.
    cam_z_origin = origins @ cam_z_row + cam_z_translation
    cam_z_direction = directions @ cam_z_row

    depth_data = depth_buffer.data.astype(np.float64)
    no_hit_mask = depth_data >= depth_buffer.no_hit_value * 0.999

    with np.errstate(divide="ignore", invalid="ignore"):
        t = (-depth_data - cam_z_origin) / cam_z_direction
    world_points = origins + t[..., None] * directions

    obj_matrix_world_inv = np.linalg.inv(np.array(volume.matrix_world, dtype=np.float64))
    local_min, local_extent = _bbox_local_min_extent(volume.bounding_box_local)

    h, w = world_points.shape[:2]
    world_points_h = np.concatenate([world_points, np.ones((h, w, 1))], axis=-1)
    local_points = world_points_h @ obj_matrix_world_inv.T
    local_points = local_points[..., :3] / local_points[..., 3:4]
    unitary_points = (local_points - local_min) / local_extent

    proj_view = np.array(yt_idv_camera.projection_matrix, dtype=np.float64) @ np.array(
        yt_idv_camera.view_matrix, dtype=np.float64
    )
    unitary_points_h = np.concatenate([unitary_points, np.ones((h, w, 1))], axis=-1)
    with np.errstate(divide="ignore", invalid="ignore"):
        clip = unitary_points_h @ proj_view.T
        ndc_z = clip[..., 2] / clip[..., 3]
    window_depth = 0.5 * ndc_z + 0.5
    window_depth = np.where(no_hit_mask, 1.0, window_depth)
    window_depth = np.nan_to_num(window_depth, nan=1.0, posinf=1.0, neginf=1.0)
    window_depth = np.clip(window_depth, 0.0, 1.0).astype("f4")

    # depth_buffer.row_order -> ordinary top-down convention (flip is its
    # own inverse, so this one check handles both directions).
    if depth_buffer.row_order == "BOTTOM_LEFT_ORIGIN":
        window_depth = window_depth[::-1, :]
    elif depth_buffer.row_order != "TOP_LEFT_ORIGIN":
        raise ValueError(f"unknown row_order {depth_buffer.row_order!r}")
    return np.ascontiguousarray(window_depth)


def _pad_to_square(depth_2d, render_size, fill_value=1.0):
    """Inverse of `_center_crop`: place a `(height, width)` array centered
    in a `(render_size, render_size)` canvas filled with `fill_value`
    elsewhere. Uses the EXACT SAME offset formula as `_center_crop`, so a
    depth value computed for Blender pixel `(y, x)` lands at the same
    square-canvas position the final color image's center-crop will later
    read pixel `(y, x)` back from -- see `_render_with_yt_idv`'s docstring
    on why this alignment is only an approximation (yt_idv renders square,
    Blender's real camera essentially never is).
    """
    height, width = depth_2d.shape
    canvas = np.full((render_size, render_size), fill_value, dtype="f4")
    y0 = (render_size - height) // 2
    x0 = (render_size - width) // 2
    canvas[y0 : y0 + height, x0 : x0 + width] = depth_2d
    return canvas


def _fix_yt_idv_axis_order(image, expected_height, expected_width):
    """Work around yt_idv bug #1 (see module docstring): `SceneGraph.image`
    returns `(width, height, 4)` for a non-square render instead of the
    conventional `(height, width, 4)`, because its own corrective
    `arr.swapaxes(0, 1)` call discards its return value and is a no-op. Fix
    it here rather than in yt_idv itself -- purely a caller-side reshape, no
    library change needed.

    Square renders are already ambiguous (both dimensions equal) and
    correct either way, so this only actually transposes when the shape
    doesn't already match what was requested.
    """
    if image.shape[:2] == (expected_height, expected_width):
        return image
    if image.shape[:2] == (expected_width, expected_height):
        return image.swapaxes(0, 1)
    raise ValueError(
        f"yt_idv returned image shape {image.shape[:2]}, matching neither "
        f"requested (height, width)=({expected_height}, {expected_width}) "
        "nor its transpose -- something beyond the known axis-order bug "
        "(see module docstring) is going on."
    )


def _to_row_order(image, row_order):
    """yt_idv's `SceneGraph.image` is already TOP_LEFT_ORIGIN (it reads the
    OpenGL framebuffer, which is bottom-first, then explicitly flips --
    `arr[::-1, :, :]` in `yt_idv.scene_graph.SceneGraph.image`). Flip again
    if the rest of this add-on's pipeline wants BOTTOM_LEFT_ORIGIN instead
    (see operators.DEPTH_ROW_ORDER, confirmed via scripts/verify_zpass.py).
    """
    if row_order == "TOP_LEFT_ORIGIN":
        return image
    if row_order == "BOTTOM_LEFT_ORIGIN":
        return image[::-1, :, :]
    raise ValueError(f"unknown row_order {row_order!r}")


def _center_crop(image, expected_height, expected_width):
    """Crop a square `(size, size, 4)` render down to `(expected_height,
    expected_width, 4)`, centered. See `yt_idv_max_intensity_renderer` for
    why the render itself is square -- this recovers the actually-requested
    (generally non-square) shape afterward.
    """
    size = image.shape[0]
    y0 = (size - expected_height) // 2
    x0 = (size - expected_width) // 2
    return image[y0 : y0 + expected_height, x0 : x0 + expected_width, :]


def _render_with_yt_idv(request: RenderRequest, configure_component) -> RenderResult:
    """Shared by both renderer modes below -- everything except how the
    `block_rendering` component itself gets configured (`configure_component`,
    called with that component right before `camera.update()`/`rc.run()`).

    Renders SQUARE (`max(height, width)` on both sides) and center-crops
    down to the actually-requested shape, rather than rendering directly at
    the requested (generally non-square, since Cycles resolutions almost
    never are) size. Workaround for a THIRD yt_idv issue, found while
    testing camera motion: non-square renders through this add-on showed a
    persistent, exact period-4 banding pattern along every row (confirmed
    via a raw scanline dump: `[value, 0, 0, value]` repeating, unaffected by
    `sample_factor` up to 10x -- ruling out a ray-marching under-sampling
    explanation) that a square render of the same scene does NOT show at
    all (confirmed directly, same camera/dataset, only the aspect changed).
    Root cause not identified (candidates: EGL surface/multisample config,
    a width/height mixup somewhere in yt_idv's own viewport or
    grid-expansion geometry shader for non-square viewports specifically --
    being investigated independently). Rendering square and cropping avoids
    it entirely at the cost of some wasted GPU work on the cropped-away
    margin and a FOV that's an approximation of the true non-square framing
    (see `_registered_camera_kwargs` -- aspect_ratio passed to yt_idv's
    camera is 1.0 here, not the request's real aspect, so the cropped
    result is a center crop of a square-FOV render, not a from-scratch
    non-square-FOV one). Consistent with this module's existing "camera
    correspondence is already approximate" stance -- see module docstring.

    Also builds and applies the external depth clip against Blender's real
    Cycles Z-buffer every call (`_cycles_depth_to_yt_idv_window_depth` +
    `set_external_depth_clip`, padded onto the square canvas via
    `_pad_to_square` -- same centering as the eventual crop, so the two
    align).
    """
    height, width = request.depth.data.shape
    render_size = max(height, width)
    rc, scene, ds = _get_context(DEFAULT_BACKEND, render_size, render_size)

    for component in scene.components:
        configure_component(component)
    scene.camera.update(
        **_registered_camera_kwargs(request.camera, request.volume, ds, render_size, render_size)
    )
    # Real bug, found and fixed while building the real end-to-end
    # sphere-and-volume demo: `TrackballCamera._compute_matrices` (the
    # method `update()`'s own trait-change observer is supposed to call) is
    # a no-op override -- `update()` alone sets the position/focus/up/fov/
    # near_plane/far_plane TRAITS correctly, but never actually recomputes
    # `projection_matrix`/`view_matrix` from them. yt_idv's OWN internal
    # code (`scene_graph._update_scene_camera_for_geometry`) always follows
    # `camera.update(...)` with an explicit `camera._update_matrices()` for
    # exactly this reason -- we were missing that second call. Without it,
    # `projection_matrix` silently keeps whatever value it held from the
    # very first time it was computed (effectively `add_scene`'s own
    # default fov/near/far), while `view_matrix` "accidentally" still
    # looked right because its own `@traitlets.default` happens to compute
    # lazily from CURRENT position/focus/up on first read after `update()`
    # -- so every previous render in this project had the correct camera
    # POSITION/ORIENTATION but the WRONG (stale-default) field of view and
    # near/far planes the whole time, without it being visually obvious.
    # Confirmed directly: `scene.camera.projection_matrix` compared against
    # `get_perspective_matrix(fov, aspect_ratio, near_plane, far_plane)`
    # mismatched before this call, matched exactly after it.
    scene.camera._update_matrices()

    window_depth = _cycles_depth_to_yt_idv_window_depth(request, scene.camera)
    set_external_depth_clip(scene, _pad_to_square(window_depth, render_size))

    image = np.asarray(rc.run(), dtype=np.float32)  # nominally (render_size, render_size, 4), [0, 1]
    image = _fix_yt_idv_axis_order(image, render_size, render_size)  # defensive; square shape makes this a no-op
    image = _to_row_order(image, request.depth.row_order)
    image = _center_crop(image, height, width)

    return RenderResult(
        rgba=np.ascontiguousarray(image),
        alpha_mode="PREMULTIPLIED",  # approximation -- see module docstring
        color_space="LINEAR",
    )


def _configure_max_intensity(component):
    component.render_method = "max_intensity"


def _apply_transfer_function_settings(component, tf_min, tf_max, tf_data):
    """The pure-Python half of `_configure_transfer_function` -- everything
    except actually building `tf_data` (which needs `yt` importable, see
    `_get_isocontour_transfer_function_data`). Split out so this part is
    unit-testable with a plain fake array/component (see
    tests/test_yt_idv_renderer.py) without needing `yt` at all.
    """
    component.tf_min = tf_min
    component.tf_max = tf_max
    component.tf_log = True
    component.transfer_function.data = tf_data


def _configure_transfer_function(component):
    component.render_method = "transfer_function"
    _apply_transfer_function_settings(
        component, TF_VALUE_MIN, TF_VALUE_MAX, _get_isocontour_transfer_function_data(),
    )


def yt_idv_max_intensity_renderer(request: RenderRequest) -> RenderResult:
    """`RENDERER_MODES["YT_IDV_MAX_INTENSITY"]` -- see module docstring for
    exactly what is and isn't real here yet.
    """
    return _render_with_yt_idv(request, _configure_max_intensity)


def yt_idv_transfer_function_renderer(request: RenderRequest) -> RenderResult:
    """`RENDERER_MODES["YT_IDV_TRANSFER_FUNCTION"]` -- a real transfer
    function (isocontour layers at fixed densities in log space -- see the
    module docstring's "Transfer function" section for `TF_VALUE_MIN`/`MAX`'s
    units and how they were found), not a placeholder colormap. Everything
    else (camera mapping, depth-clip against Cycles' Z-buffer,
    square-render-and-crop) is exactly as in `yt_idv_max_intensity_renderer`
    -- see its and the module's docstrings.
    """
    return _render_with_yt_idv(request, _configure_transfer_function)


def _configure_projection(component):
    component.render_method = "projection"


def yt_idv_projection_renderer(request: RenderRequest) -> RenderResult:
    """`RENDERER_MODES["YT_IDV_PROJECTION"]` -- unweighted integration along
    the line of sight (yt_idv's `"projection"` render_method: additive
    `GL_FUNC_ADD` blending, not `max_intensity`'s `GL_MAX`) -- tends to look
    smoother/less speckled than MIP for the same data, since it's summing
    contributions along the ray rather than keeping only the single
    brightest sample. Everything else is exactly as in
    `yt_idv_max_intensity_renderer` -- see its and the module's docstrings.
    """
    return _render_with_yt_idv(request, _configure_projection)
