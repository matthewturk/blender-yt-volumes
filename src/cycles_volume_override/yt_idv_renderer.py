"""Stub external renderer that calls the REAL yt_idv volume renderer
(offscreen OpenGL) instead of the toy/debug renderers in
`sample_renderers.py`/`external_render.py`. Two render modes:
`yt_idv_max_intensity_renderer` (Maximum Intensity Projection) and
`yt_idv_transfer_function_renderer` (a real transfer function -- isocontour
layers at fixed densities in log space, the standard first cut per the
brief this was extended against). Both share `_render_with_yt_idv` for
everything except how the `block_rendering` component itself is configured.

Deliberately scoped down, per the brief this was written against:

  - Renders one hardcoded sample dataset (`yt.load_sample("IsolatedGalaxy")`)
    by default, NEVER `request.volume.dataset.dataset_path`/`dataset_kind`
    directly -- those two generic fields still aren't wired up here (a bare
    path/kind string can't express field selection, units, derived fields,
    cut regions, or any of the other real yt setup a dataset actually needs).
    Instead, `custom_params["yt_idv_volume_provider"]` -- a
    `"module.submodule:function_name"` string pointing at a Python function
    YOU write -- lets you replace the hardcoded sample with arbitrary real
    yt/yt_idv setup, called once per object per frame, kept stateful across
    calls. See "Custom volume setup" below for the full harness.
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

**Threshold fog transfer function, plus a THIRD real yt_idv bug found and
fixed while building it.** User request, verbatim: "a new colormap that is
fully transparent except at some threshold value... ramp up the alpha to
make it look foggy" -- for "partly-occluded stuff" like the demo above, but
with a real, continuous, tunable density-vs-opacity relationship instead of
either narrow isocontour lines or an uncontrolled colormap opacity floor.
See `_build_threshold_fog_transfer_function_data`'s own docstring for the
shape and why it needed no new shader/uniform work at all
(`transfer_function.frag.glsl` already discards anything below `tf_min`
entirely -- setting `tf_min` to the threshold IS the whole "fully
transparent below this" mechanism, for free) and `TF_FOG_RAMP_POWER`'s own
comment for how the actual density-vs-alpha curve was tuned against a real
scene (not guessed).

While verifying the FIRST real-Cycles-driven render with this new mode, the
background composited as solid black instead of transparent -- confirmed
via direct pixel inspection to be a real yt_idv bug, NOT a repeat of bug 5
above (already fixed): `passthrough.frag.glsl` -- used as `block_rendering.
transfer_function`'s own SECOND pass, per `shaders/shaderlist.yaml` --
unconditionally sets `color.a = 1.0`, discarding the FIRST pass's real,
per-pixel accumulated alpha (`transfer_function.frag.glsl`'s own physically
-based front-to-back opacity integral) entirely. This has presumably always
been true of the pre-existing isocontour transfer function mode too --
never specifically alpha-checked before now, since `render_real_demo.py`'s
own background-alpha verification has only ever been run against
`max_intensity`/`projection` (whose `apply_colormap` second pass correctly
preserves the first pass's real alpha), not `transfer_function`. NOT fixed
by editing `passthrough.frag.glsl` in place -- that shader is shared by
several genuinely-opaque-by-design usages (outlines, text, line plots,
curves; see `shaderlist.yaml`'s other `second_fragment: passthrough`
entries), so changing it globally risked regressing those. Fixed instead by
adding a new, minimal sibling shader, `passthrough_alpha.frag.glsl`
(identical except it omits the `color.a = 1.0` line), and pointing ONLY
`block_rendering.transfer_function`'s `second_fragment` (both the cartesian
and octree geometry variants) at it -- see `scripts/yt_idv_exploration/
yt_idv_external_depth_clip.patch` for the exact diff. This same fix also
benefits the pre-existing isocontour `YT_IDV_TRANSFER_FUNCTION` mode's
background-alpha correctness, not just the new fog mode.

**A separate, real re-tuning need, NOT a new bug, surfaced by this same
investigation:** this module's own earlier camera-matrix bug fix (see
"Two real bugs in THIS ADD-ON's own code" above) changed the actual
geometry every yt_idv render uses -- correctly -- but that also shifts
which density values a given camera/scene combination samples along its
rays. `transfer_function.frag.glsl`'s accumulation is sensitive to that in
a way `max_intensity`/`projection` aren't (see `TF_FOG_RAMP_POWER`'s own
comment for the full mechanism) -- confirmed the PRE-EXISTING isocontour
transfer function's `TF_VALUE_MIN`/`MAX` (tuned against the OLD, buggy
camera) now produce essentially no visible signal at all against the
corrected camera. Re-tuned `TF_FOG_VALUE_MIN`/`MAX`/`RAMP_POWER` for the new
fog mode against a real scene; `TF_VALUE_MIN`/`MAX` (the older isocontour
mode) were NOT re-tuned here -- flagged as a follow-up, see README
"What's next".

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
  - The offscreen GL context (`_get_render_context`), the hardcoded sample
    dataset (`_load_sample_source`), and every per-object volume-provider
    cache (`_volume_scene_cache`/`_render_result_cache`, see "Custom volume
    setup" below) are all cached at module level and never explicitly torn
    down -- there's no hook into `operators.unregister()` for any of this
    yet, so disabling/reloading the add-on leaks the GL context (and holds
    onto whatever a custom provider itself cached) until Blender itself
    exits. Fine for exploring; worth a real fix before this is anything
    more than that.
  - Every module constant below that shapes a render (backend, both
    transfer functions' value ranges/colormaps/ramp shapes, MIP-alpha's
    colormap/max/ramp) is now ALSO a real, live Preferences-panel field --
    see "Preferences panel / experimentation" below for the override
    mechanism and exactly which constant each field maps to. The module
    constants themselves remain the fallback DEFAULTS (used whenever a
    caller doesn't set the corresponding `custom_params` key -- every
    exploration script in `scripts/yt_idv_exploration/` that builds its own
    `RenderRequest` by hand still gets these, unaffected) -- and they're
    still tuned by trial and error for IsolatedGalaxy's density field (see
    "Transfer function" above), so they won't transfer to a different
    dataset/field without re-tuning, same caveat as before, just no longer
    requiring a source edit to act on.

**Preferences panel / experimentation.** Every yt_idv-specific rendering
knob above (backend + both transfer functions' value ranges/colormaps/ramp
shapes + MIP-alpha's colormap/max/ramp -- deliberately NOT dataset
selection, which is still hardcoded to `SAMPLE_DATASET_NAME`/`SAMPLE_FIELD`,
out of scope until dataset loading itself is designed) is exposed as a real
field in `VolumeOverridePreferences.yt_idv` (a `YtIdvRenderSettings`
`PointerProperty`, see `operators.py`), drawn in the add-on's own
Preferences panel ONLY when `renderer_mode` is one of the `YT_IDV_*`
options -- selecting `STUB`/`DEBUG_FLAT`/`SAMPLE_CONSTANT` hides the whole
section, so the panel stays uncluttered for non-yt_idv renderers. These
settings reach this module the SAME way any renderer-specific knob is
supposed to cross the `external_render.py` boundary: `operators.
build_volume_params` merges them into `VolumeObjectParams.custom_params`
(see `operators._yt_idv_custom_params`) under `yt_idv_*`-prefixed keys,
and each `_configure_*` function here reads `custom_params.get(key,
MODULE_CONSTANT)` -- so this module still never imports `bpy` or knows
Preferences exist, and a non-Blender caller building its own `RenderRequest`
(every exploration script) is completely unaffected, getting the module
constants exactly as before.

**Curve-driven alpha / image-driven color (threshold fog + MIP-alpha only).**
The idea floated in this docstring's previous revision -- design the
transfer function visually instead of via `ramp_power`/a colormap name --
is now real, for the two CONTINUOUS-ramp modes (`YT_IDV_THRESHOLD_FOG`/
`YT_IDV_MAX_INTENSITY_ALPHA`; NOT the isocontour `YT_IDV_TRANSFER_FUNCTION`,
whose `add_layers()`-based shape doesn't reduce to "one alpha-vs-position
curve" the same way). Two independent, optional overrides, either or both
active at once:

  - Alpha: instead of `alpha_max * frac**ramp_power` (`_threshold_fog_
    alpha_ramp`), an arbitrary user-drawn curve -- `operators.py` hosts a
    real `bpy.types.CurveMapping` (not a Material bake; a `CurveMapping`
    can't be a `PointerProperty` on a plain PropertyGroup at all --
    confirmed: `TypeError: PointerProperty(...) expected an RNA type
    derived from ID or ID Property Group` -- so it's piggybacked on a
    hidden `ShaderNodeFloatCurve` node's own native `.mapping` instead,
    drawn directly in this add-on's Preferences panel via `UILayout.
    template_curve_mapping(node, "mapping")`, the same widget a Shader
    Editor's N-panel would show) evaluated at N sample points and passed
    through as a plain list of floats, `custom_params["yt_idv_alpha_curve"]`
    -- see `operators._evaluate_alpha_curve`/`_get_or_create_alpha_curve_
    node`.
  - Color: instead of a named yt/matplotlib colormap, an arbitrary Blender
    `Image` datablock, read back and treated as "functionally 1D" per the
    user's own framing -- averaged across all rows (collapsing whatever
    vertical variation exists) then resampled horizontally, so ANY image
    works regardless of its native resolution/aspect, not just a literal
    1-pixel-tall gradient strip -- passed through as a nested list,
    `custom_params["yt_idv_color_image_rgba"]`. See `operators.
    _sample_image_as_1d_rgba`.

Both are resampled to whatever `nbins` the LUT actually needs
(`_resample_1d`/`_resample_rgb`, plain `np.interp`) rather than assumed to
already be the right length -- decouples this module from whatever sample
count the curve was evaluated at or the image's own native width. Alpha and
color are deliberately INDEPENDENT switches (`_build_threshold_fog_
transfer_function_data`/`_build_mip_alpha_colormap_data` take `alpha_curve`
and `color_rgba` as two separate optional parameters) -- you can mix a
hand-drawn alpha curve with a named colormap's color, or a `ramp_power`
alpha with an image-driven color, or any other combination; whichever one
is `None` falls back to the pre-existing mechanism for just that channel.
Alpha itself NEVER comes from the image (even if it has real alpha data) --
one mechanism answers "how transparent here", not two competing ones; see
these functions' own docstrings.

Not independently visually verified end-to-end in this sandbox: there is no
interactive display available here to confirm `template_curve_mapping`
actually renders/responds as an editable widget inside a real Preferences
panel (as opposed to merely being a valid, non-crashing RNA call) -- the
underlying data path (CurveMapping creation/evaluation, Image pixel
readback, resampling, LUT construction, cache-key correctness) WAS verified
directly by running real Blender code and checking the numbers. Re-verify
the widget itself interactively before relying on it for real work.

**Custom volume setup (the `yt_idv_volume_provider` harness).** Requested
directly: a well-defined way to hand this module arbitrary real yt setup
(load a real dataset, pick a field, build a cut region/derived field,
anything yt can do) instead of the hardcoded `IsolatedGalaxy` sample, without
this module needing to know anything about YOUR specific dataset. The shape
of the problem: loading a real dataset is often the single most expensive
thing that happens (disk I/O, hierarchy parsing, sometimes gigabytes), so
whatever hook exists has to let the caller be as stateful/lazy as
`_load_sample_source`'s own "cache once per Blender session" already is --
a plain "load this path every render call" function would make the add-on
unusable on anything but a toy dataset.

The mechanism: `custom_params["yt_idv_volume_provider"]`, a
`"module.submodule:function_name"` string (see `properties.
VolumeOverrideProperties.yt_idv_volume_provider` for the per-object UI field
that sets it) resolved via `importlib` (`_resolve_volume_provider`) to a
plain Python callable, called as `provider(ctx) -> VolumeSetupResult`, once
per flagged object per frame, from `_get_volume_scene` (itself called by
`_render_with_yt_idv`, so every yt_idv renderer mode gets this uniformly,
not just one). `ctx` (`VolumeSetupContext`) is everything the provider
needs to either build something new or confirm the old one is still good:

  - `ctx.render_context`: the LIVE, already-created-and-sized `yt_idv`
    render context (`rc`) for this object's current (backend, render size)
    -- call `ctx.render_context.add_scene(data_source, field, ...)`
    yourself to build a scene; this module never calls `add_scene` on your
    behalf, so you have full control over the field name, `no_ghost`, or
    anything else `add_scene` accepts.
  - `ctx.previous_scene`/`ctx.previous_ds`: EXACTLY what your own last call
    returned as `scene`/`ds` (see `VolumeSetupResult` below) -- `None` on
    the very first call for this object, or after this module itself
    detects the cached scene is stale (the render context it belonged to
    no longer matches, e.g. the backend or render resolution changed, in
    which case a rebuild is unavoidable regardless of your own dataset
    state, so `previous_scene` comes back `None` even if your dataset is
    still perfectly valid). This is how "stateful, don't redo work you
    don't have to" actually works: the common case is `if ctx.previous_
    scene is not None: return VolumeSetupResult(ctx.previous_scene, ctx.
    previous_ds, changed=False)` -- most calls, most frames, should hit
    this and do nothing else.
  - `ctx.dataset`: this object's `DatasetParams` -- `dataset_path`/
    `dataset_kind` (whatever you typed on the object, still yours to
    interpret however you like), `requested_time` (see that field's own
    docstring for the frame->time mapping), and `version.hash`/`.changed`
    (Blender's own convenience signal for "did dataset_path/dataset_kind/
    the object's identity change since last render" -- see VersionInfo's
    own docstring for why `.changed` is a hint, not authoritative, and
    `.hash` is the one safe to actually cache against).
  - `ctx.custom_params`: the full `custom_params` dict (same one
    `configure_component` receives) -- your own extra settings, if you
    added any via `custom_params_extra`, are right there under whatever key
    you gave them.
  - `ctx.frame`: `FrameParams` (`frame_number`/`scene_time_seconds`), for
    providers that want to key off Blender's frame directly rather than
    `dataset.requested_time`'s mapped value.

`VolumeSetupResult(scene, ds, changed)`: `scene` is what `_render_with_yt_idv`
actually renders (its own `.components` get configured, its own `.camera`
gets positioned) -- normally `ctx.render_context.add_scene(...)`'s own
return value, or `ctx.previous_scene` unchanged. `ds` needs only
`.domain_center` (a real yt `Dataset`, or any object/array with that
attribute -- see `_registered_camera_kwargs`/`_domain_array_unitary`) --
used purely as a reference point for a sensible look-at distance, not for
positioning itself. `changed`: **True** means the underlying volume DATA
itself changed since your last call (new dataset, new timestep, a rebuilt
cut region, anything that would make a stale render visibly wrong) --
`_render_with_yt_idv` treats this as "you must re-render this frame,
period," regardless of whether the camera moved. **False** means nothing on
your end changed -- `_render_with_yt_idv` then checks its OWN independent
signal (has the camera, the Cycles depth buffer, or any shading knob in
`custom_params` changed since the last frame THIS object was actually
rendered -- a content hash via `change_tracking.version_for`, same
authoritative-hash-not-just-a-flag discipline as `VersionInfo` elsewhere in
this add-on) and skips the entire `rc.run()`/depth-clip/compositing dance,
returning the previous frame's own `RenderResult` verbatim, ONLY if that
signal also says nothing changed. Get `changed` wrong in the "should have
been True" direction and a real data change silently doesn't show up on
screen; get it wrong the other way (always `True`) and you simply lose the
skip-render optimization -- still correct, just slower. When genuinely
unsure, returning `True` is the safe default.

Not wired to anything if `custom_params` doesn't set `"yt_idv_volume_
provider"` at all -- `_default_volume_provider` (the pre-existing hardcoded-
IsolatedGalaxy behavior, reimplemented against this exact same protocol so
it isn't a special case) is what every flagged object gets, unchanged,
until you opt in per-object. Multiple flagged objects sharing one provider
function, or each pointing at a different one, both work today, dispatched
per-object via `RenderRequest.volume.dataset.dataset_id` (the same stable
cache key `change_tracking.py` already uses elsewhere) -- INCLUDING when
several such objects share one render_context (same backend/render size,
the common case, since resolution is scene-wide): `_render_with_yt_idv`
re-asserts `rc.scene = scene` on every call before `rc.run()`, a real,
necessary fix rather than an assumption -- `yt_idv.rendering_contexts.
base_context.BaseContext.add_scene` only ever mutates `self.scene`, and
`OffscreenRenderingContext.run()` renders THAT, not any particular Python
`scene` reference this module might be holding -- so without re-asserting
it, whichever object's `add_scene()` ran most recently (possibly a
DIFFERENT object, from this same frame or an earlier one) would silently
own every subsequent `rc.run()` on a shared context, even for an object
whose own cached scene never needed rebuilding at all. Each object still
renders and Alpha-Over composites fully independently, exactly like today
-- genuinely compositing MULTIPLE volumes into one COMBINED yt_idv scene,
sampled together in a single ray march, is a separate, harder problem, not
attempted here.

Verified directly, not just designed on paper: `scripts/yt_idv_exploration/
blender_scene/custom_volume_provider_demo.py` defines a real provider
function (in its own module, `_custom_volume_provider_example.py` -- loads
the same `IsolatedGalaxy` sample explicitly, through this exact mechanism
instead of the hardcoded default) and confirms, against real renders,
sharing one real render_context with the default provider throughout: (1)
the custom provider's output matches the default provider's own, pixel for
pixel; (2) the provider is NOT called with `previous_scene=None` on its
second call at the same render size (it was actually reused, not silently
rebuilt); (3) an IDENTICAL second call (same camera, same depth, same
params, `changed=False`) returns the exact same `RenderResult` object back
(the skip-cache actually skips `rc.run()`, not just "happens to produce the
same pixels"), while a camera-only change with `changed=False` from the
provider still triggers a real, different re-render (the skip-cache doesn't
over-cache); and (4) re-rendering the DEFAULT object again, after the
CUSTOM object (sharing the same render_context) called `add_scene()` in
between, still ends up with `rc.scene` correctly pointing at the default
object's own scene, not the custom object's leftover one -- confirmed
failing without the `rc.scene = scene` re-assertion above (both this real
script AND `test_render_with_yt_idv_reasserts_rc_scene_when_objects_share_
a_context` were run with that line disabled to confirm they actually catch
its absence, not just happening to pass either way).

Still explicitly future work, unattempted here: aspect ratio (yt_idv's own
square-render/crop approximation, see `_render_with_yt_idv`'s docstring, is
unrelated to and unaffected by any of the above -- a provider cannot itself
fix the non-square FOV mismatch) and genuinely compositing multiple
providers'/objects' volumes into a single combined yt_idv scene.
"""

import dataclasses
import hashlib
import importlib
import math

import numpy as np

from . import change_tracking
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

# --- Threshold fog transfer function (fully transparent below a threshold,
# alpha ramps up above it) -- see `_build_threshold_fog_transfer_function_
# data`'s docstring for the shape and why it needed no new shader/uniform
# work at all: `transfer_function.frag.glsl` ALREADY discards any sample
# outside `[tf_min, tf_max]` entirely (`if(tex_sample < tm) return false;`),
# so setting `tf_min` to the desired threshold gives a clean, free cutoff --
# the "isocontour" and "threshold fog" transfer functions differ only in
# what LUT data they hand to the exact same `render_method="transfer_
# function"` shader path. Same [0, 1]-normalized value space as
# TF_VALUE_MIN/MAX above (see that section's docstring for why) -- reusing
# those same bounds here since they're already known-good for this
# dataset/field's useful density range.
TF_FOG_VALUE_MIN = 1e-9  # the threshold -- fully invisible below this
TF_FOG_VALUE_MAX = 1e-2  # alpha reaches TF_FOG_ALPHA_MAX at this value
TF_FOG_COLORMAP = "dusk"  # a real yt/matplotlib colormap for the fog's own color
TF_FOG_ALPHA_MAX = 0.85  # deliberately NOT 1.0 -- stays "foggy"/soft even at the densest point, not a hard solid wall
TF_FOG_RAMP_POWER = 0.15  # < 1 = alpha ramps up FAST just past the threshold, then flattens out toward TF_FOG_ALPHA_MAX -- see this constant's own re-tuning note below for why this (not TF_VALUE_MIN/MAX's old 2.0-ish intuition) is what actually works here
# `TF_FOG_VALUE_MIN`/`MAX`/`RAMP_POWER` above were re-tuned AFTER a real
# regression, worth recording: this module's own real-Cycles-driven camera
# bug fixes (`_render_with_yt_idv`'s missing `scene.camera._update_matrices()`
# call, `_near_far_planes`) changed the ACTUAL geometry every yt_idv render
# uses -- correct now, but it also shifts which density values a given
# camera/scene combination actually samples along its rays. `transfer_
# function.frag.glsl`'s accumulation is a REAL, physically-based front-to-
# back opacity integral (`dt * tf_sample.a` per ray-march step, not a
# presence/absence test) -- unlike `max_intensity`/`projection`'s own
# accumulation, which (via `apply_colormap`'s auto-computed per-view
# cmap_min/cmap_max) effectively just asks "was ANY nonzero density found
# along this ray", the transfer function mode is sensitive to the ACTUAL
# accumulated `sum(dt * alpha)` over the real path length -- so it's the one
# mode where a camera-geometry change can silently starve the total
# achievable opacity even though "the same density values are still there
# somewhere". Empirically found (see `scripts/yt_idv_exploration/
# render_partial_occlusion_demo.py`'s own investigation, sweeping value_min/
# max/ramp_power against a real cached Cycles+depth-clip request rather than
# guessing): with the corrected camera, the isocontour transfer function's
# OLD TF_VALUE_MIN/MAX (1e-6/2e-3, tuned against the OLD, buggy camera)
# produces essentially NO visible signal any more (max accumulated alpha
# under 0.004) -- not a new bug, the same "won't transfer to a different
# camera configuration without re-tuning" caveat already documented for
# TF_VALUE_MIN/MAX below, now actually triggered by this module's own later
# camera fix. A front-loaded ramp (`ramp_power < 1`, alpha rises fast just
# past threshold rather than climbing all the way to TF_FOG_VALUE_MAX
# gradually) recovers a strong, clearly visible result because it spreads
# high per-step alpha over a WIDER slice of the density values actually
# encountered, rather than concentrating it in a narrow band near the very
# top of the range -- confirmed empirically (0.15 vs 1.0 raised peak
# accumulated alpha from ~0.16 to ~0.60 on the same real scene). This same
# investigation strongly suggests TF_VALUE_MIN/MAX (the pre-existing
# isocontour mode, used by `YT_IDV_TRANSFER_FUNCTION`) is NOW similarly
# stale post-camera-fix and needs its own re-tuning pass -- not done here,
# flagged as a follow-up (see README "What's next").

# --- Maximum Intensity with a value-driven alpha (transparent at the low
# end of the per-view auto-scaled range, opaque at the high end) -- see
# `apply_colormap_alpha.frag.glsl`'s own docstring for why plain
# `max_intensity` can't do this at all (its own alpha is a binary "was
# anything found" flag, completely decoupled from the selected value).
# Unlike TF_FOG_VALUE_MIN/MAX above, this rides on `max_intensity`'s
# EXISTING per-view auto cmap_min/cmap_max (`BaseComponent.
# _reset_cmap_bounds`) rather than fixed dataset-tuned bounds -- the ramp
# below is defined purely as a FRACTION of whatever that auto-computed
# range turns out to be for the current camera, so it needs no dataset-
# specific recalibration the way TF_FOG_VALUE_MIN/MAX did.
MIP_ALPHA_COLORMAP = "dusk"
MIP_ALPHA_MAX = 0.7  # NOT 1.0 -- see render_zoomed_core_demo.py's own docstring: at 1.0, the volume's own near-camera material builds up to full opacity before the ray even reaches an embedded marker object, hiding it completely via Alpha-Over regardless of the object's own real depth-clip visibility. Capped below 1.0 so SOMETHING behind the densest visible material always shows through.
MIP_ALPHA_RAMP_POWER = 0.15  # < 1 = ramps up quickly past the low end, same reasoning as TF_FOG_RAMP_POWER -- 0.15 chosen empirically against a real zoomed-in-on-the-core scene (see scripts/yt_idv_exploration/blender_scene/render_zoomed_core_demo.py), sweeping 0.15/0.3/0.6/1.0: lower values keep more of the diffuse halo visibly (if faintly) present instead of vanishing to fully transparent, without ever losing the transparent-low/opaque-high shape itself
MIP_ALPHA_NBINS = 256

_dataset_cache = {}  # dataset name -> (ds, data_source) -- _default_volume_provider's own cache, same role as before
_render_context_cache = {}  # (backend, width, height) -> rc -- renamed from _context_cache now that scene/dataset setup is a per-object concern (see _volume_scene_cache) rather than tied 1:1 to the GL context
_volume_scene_cache = {}  # dataset_id -> (scene, ds, render_context_key) -- one flagged object's own last-built (or reused) yt_idv Scene; render_context_key lets _get_volume_scene tell a stale-context scene apart from a genuinely-unchanged one
_render_result_cache = {}  # dataset_id -> RenderResult -- the last successful render for this object, returned as-is when _render_signature_changed says nothing that would affect the image has changed
_transfer_function_cache = {}  # (n_layers, colormap, value_min, value_max) -> (256, 1, 4) uint8 RGBA texture data
_fog_transfer_function_cache = {}  # (colormap, value_min, value_max, alpha_max, ramp_power) -> (256, 1, 4) uint8 RGBA texture data
_mip_alpha_colormap_cache = {}  # (colormap, alpha_max, ramp_power) -> (256, 4) float32 RGBA colormap data


def _load_sample_source():
    """Lazily load the hardcoded sample dataset ONCE per Blender session.

    This is the expensive part (disk I/O + hierarchy parsing) -- the
    two-pass pipeline calls the render function once per flagged object per
    frame, so caching here isn't a nice-to-have, it's required for this to
    be usable at all across more than a single call. Used only by
    `_default_volume_provider` now -- a custom `yt_idv_volume_provider` is
    free to load however/whatever it wants instead (see "Custom volume
    setup" above).
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


def _get_render_context(backend, width, height):
    """Lazily create ONE offscreen render context per (backend, width,
    height), reused across calls AND across every flagged object that
    shares those three values -- yt_idv's offscreen contexts fix their
    framebuffer size at creation, so a resolution change (e.g. the user
    resizes the render) transparently gets its own cached context rather
    than reusing a wrongly-sized one.

    Narrower than the old (pre-`yt_idv_volume_provider`) `_get_context`,
    which also built the ONE hardcoded scene right here -- scene/dataset
    setup is now a per-object concern (`_get_volume_scene`, dispatched
    through whatever provider `custom_params["yt_idv_volume_provider"]`
    names), since different flagged objects sharing this same render
    context may legitimately want different volumes.
    """
    key = (backend, width, height)
    cached = _render_context_cache.get(key)
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
    rc = yt_idv.render_context(backend, width=width, height=height)
    _render_context_cache[key] = rc
    return rc


@dataclasses.dataclass
class VolumeSetupContext:
    """Passed to a `custom_params["yt_idv_volume_provider"]` callable, once
    per flagged object per frame -- see module docstring "Custom volume
    setup" for the full contract and rationale.
    """

    render_context: "object"  # the live yt_idv render context (`rc`) -- call .add_scene(...) on this yourself
    dataset: "object"  # external_render.DatasetParams for this object -- dataset_path/dataset_kind/requested_time/version, whatever you put there
    custom_params: dict  # same dict configure_component() receives -- your own custom_params_extra entries included
    frame: "object"  # external_render.FrameParams
    previous_scene: "object | None"  # exactly what YOUR last call returned as `scene`, or None (first call, or the render context changed underneath it)
    previous_ds: "object | None"  # exactly what YOUR last call returned as `ds`, or None alongside previous_scene


@dataclasses.dataclass
class VolumeSetupResult:
    """Returned by a `yt_idv_volume_provider` callable -- see
    `VolumeSetupContext`/module docstring "Custom volume setup".
    """

    scene: "object"  # what _render_with_yt_idv actually renders (.components get configured, .camera gets positioned)
    ds: "object"  # needs only `.domain_center` -- see _registered_camera_kwargs/_domain_array_unitary
    changed: bool  # True: the volume DATA itself changed, force a re-render regardless of camera. False: nothing on your end changed -- a re-render still happens if the camera/depth/shading knobs did, but can be skipped entirely if nothing did.


def _resolve_volume_provider(spec):
    """`spec` is `"module.submodule:function_name"` -- the exact string a
    user puts in `custom_params["yt_idv_volume_provider"]` (see
    `properties.VolumeOverrideProperties.yt_idv_volume_provider`). Raises
    loudly (ValueError/ImportError/AttributeError) on anything malformed
    rather than silently falling back to the default provider -- a
    misconfigured provider should fail obviously at render time, not
    quietly render the wrong (or hardcoded-sample) volume.
    """
    if not isinstance(spec, str) or ":" not in spec:
        raise ValueError(
            "custom_params['yt_idv_volume_provider'] must be a "
            f"'module.submodule:function_name' string, got {spec!r}"
        )
    module_path, _, func_name = spec.rpartition(":")
    module = importlib.import_module(module_path)
    try:
        return getattr(module, func_name)
    except AttributeError as exc:
        raise AttributeError(
            f"module {module_path!r} has no attribute {func_name!r} "
            f"(from custom_params['yt_idv_volume_provider']={spec!r})"
        ) from exc


def _default_volume_provider(ctx: VolumeSetupContext) -> VolumeSetupResult:
    """What every flagged object gets when `custom_params` doesn't set
    `"yt_idv_volume_provider"` -- the pre-existing hardcoded-IsolatedGalaxy
    behavior, reimplemented against the exact same protocol a real custom
    provider uses, rather than as a special case `_get_volume_scene` has to
    know about separately. Loads (`_load_sample_source`, module-level
    `_dataset_cache`) at most ONCE per Blender session and never reports
    `changed=True` again after that -- there is nothing about this fixed
    sample that ever changes.
    """
    if ctx.previous_scene is not None:
        return VolumeSetupResult(scene=ctx.previous_scene, ds=ctx.previous_ds, changed=False)
    ds, data_source = _load_sample_source()
    scene = ctx.render_context.add_scene(data_source, SAMPLE_FIELD, no_ghost=True)
    return VolumeSetupResult(scene=scene, ds=ds, changed=True)


def _get_volume_scene(request: RenderRequest, render_size: int):
    """Resolve (via `custom_params["yt_idv_volume_provider"]`, falling back
    to `_default_volume_provider`) and cache the yt_idv Scene + yt Dataset
    for ONE flagged object, calling the provider at most once per frame.
    See module docstring "Custom volume setup" for the full harness this
    implements.

    Returns `(rc, scene, ds, changed)` -- `rc` and `scene` are what
    `_render_with_yt_idv` renders with; `changed` is the provider's own
    signal (or True/False for `_default_volume_provider`'s first/later
    calls), which `_render_with_yt_idv` combines with its own
    camera/depth/params change signal to decide whether `rc.run()` is
    actually needed this frame.
    """
    custom_params = request.volume.custom_params or {}
    backend = custom_params.get("yt_idv_backend", DEFAULT_BACKEND)
    context_key = (backend, render_size, render_size)
    rc = _get_render_context(*context_key)

    provider_spec = custom_params.get("yt_idv_volume_provider")
    provider = _resolve_volume_provider(provider_spec) if provider_spec else _default_volume_provider

    dataset_id = request.volume.dataset.dataset_id
    cached = _volume_scene_cache.get(dataset_id)
    if cached is not None and cached[2] == context_key:
        previous_scene, previous_ds = cached[0], cached[1]
    else:
        # No cached scene at all, OR it belongs to a different (backend,
        # render_size) render context -- e.g. the output resolution or
        # backend changed -- and can't be reused regardless of whether the
        # provider's OWN dataset state is still fine. The provider still
        # gets called (not bypassed) so it can decide for itself how much
        # of its own state (a loaded dataset vs. the yt_idv Scene wrapping
        # it) actually needs rebuilding.
        previous_scene, previous_ds = None, None

    ctx = VolumeSetupContext(
        render_context=rc,
        dataset=request.volume.dataset,
        custom_params=custom_params,
        frame=request.frame,
        previous_scene=previous_scene,
        previous_ds=previous_ds,
    )
    result = provider(ctx)
    _volume_scene_cache[dataset_id] = (result.scene, result.ds, context_key)
    return rc, result.scene, result.ds, result.changed


def _render_signature(request: RenderRequest, render_size: int) -> dict:
    """Everything besides the volume DATA itself that affects the rendered
    image: camera, the real Cycles depth buffer, and every shading knob in
    `custom_params` (via `params_version.hash`, which ALREADY covers both
    the per-object fields like density_scale and the yt_idv-Preferences-
    derived `yt_idv_*` keys -- see `operators.build_volume_params`, which
    hashes `custom_params` only after merging both in). JSON-serializable
    by construction (`dataclasses.asdict` on a plain-data CameraParams,
    everything else already a plain str/int) -- see
    `change_tracking._stable_hash`'s own requirement.

    The depth buffer is hashed by content (`.tobytes()`), not compared by
    identity or shape alone -- a real Cycles occluder moving elsewhere in
    the scene changes this same buffer without necessarily changing the
    camera at all, and that genuinely needs a fresh render too (the whole
    point of the depth-clip feature is that a solid object depth-clips the
    volume; if the occluder is what moved, the volume's own rendered
    silhouette needs to move with it, even from an unmoving camera).
    """
    return {
        "camera": dataclasses.asdict(request.camera),
        "depth_shape": list(request.depth.data.shape),
        "depth_hash": hashlib.sha256(np.ascontiguousarray(request.depth.data).tobytes()).hexdigest(),
        "render_size": render_size,
        "params_hash": request.volume.params_version.hash,
    }


def _render_signature_changed(dataset_id: str, request: RenderRequest, render_size: int, volume_changed: bool) -> bool:
    """True if `_render_with_yt_idv` actually needs to call `rc.run()` this
    frame -- either the volume provider itself reported `changed=True`
    (checked first, short-circuiting: no need to even compute the
    signature), or `_render_signature`'s own content hash differs from the
    last frame THIS object was actually rendered (see `change_tracking.
    version_for` -- same authoritative-hash, not-just-a-flag discipline as
    `VersionInfo` elsewhere in this add-on). A signature is recorded on
    every call, including `volume_changed=True` ones, so the NEXT
    nothing-changed frame has an accurate baseline to compare against.
    """
    signature = _render_signature(request, render_size)
    version = change_tracking.version_for(dataset_id, "yt_idv_render_signature", signature)
    return volume_changed or version.changed


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


def _get_isocontour_transfer_function_data(value_min, value_max, n_layers, colormap):
    """Cached -- building the transfer function itself is cheap (milliseconds),
    but there's no reason to redo it every single frame when the settings
    haven't changed. Takes explicit values (rather than reading `TF_VALUE_
    MIN`/`MAX`/`TF_N_LAYERS`/`TF_COLORMAP` directly) so callers can override
    them -- see `_configure_transfer_function`'s own docstring for where
    those overrides actually come from (the add-on's Preferences panel, via
    `VolumeObjectParams.custom_params`).
    """
    key = (n_layers, colormap, value_min, value_max)
    cached = _transfer_function_cache.get(key)
    if cached is not None:
        return cached
    data = _build_isocontour_transfer_function_data(value_min, value_max, n_layers, colormap)
    _transfer_function_cache[key] = data
    return data


def _threshold_fog_alpha_ramp(nbins, alpha_max, ramp_power):
    """The actual "fully transparent below a threshold, alpha ramps up
    above it" shape, as a plain `(nbins,)` array in `[0, alpha_max]` --
    split out from `_build_threshold_fog_transfer_function_data` so this
    part is unit-testable with plain numpy, no `yt`/matplotlib needed (same
    reasoning as `_apply_transfer_function_settings` vs `_configure_
    transfer_function`).

    `frac = linspace(0, 1, nbins)` is the ramp's position along the LUT,
    bin 0 -> `TF_FOG_VALUE_MIN` (the threshold) and bin `nbins-1` ->
    `TF_FOG_VALUE_MAX` -- `_apply_transfer_function_settings` always sets
    `tf_log = True`, so `transfer_function.frag.glsl` itself takes the log
    of the sampled density AND of `tf_min`/`tf_max` before mapping into
    this same `[0, 1]` bin-position space (`tex_sample_norm.x`), meaning
    `frac` is uniform in LOG-density space, not raw linear density -- this
    function doesn't need its own `math.log` anywhere, it just builds "the
    array indexed by bin position" and the shader's existing log transform
    handles the rest, exactly as `_build_isocontour_transfer_function_data`
    already relies on for its own bin placement. So `frac ** power`
    IS the whole ramp shape: `power == 1.0` is a plain linear ramp (in log
    space);
    `power > 1.0` stays near-zero for longer before climbing (fog builds up
    slowly, then thickens quickly near the top -- the requested "ramp up
    the alpha to make it look foggy" effect); `power < 1.0` front-loads the
    climb instead (thick almost immediately past the threshold, flattening
    out toward `alpha_max`). Bin 0 itself is still driven to exactly 0.0
    regardless of `power` (`0.0 ** anything == 0.0`), matching the "fully
    transparent below threshold" requirement -- though note `transfer_
    function.frag.glsl` already discards anything BELOW `tf_min` entirely
    via its own `tex_sample < tm` check, so this array's own bin-0 value is
    belt-and-suspenders, not load-bearing for that specific guarantee.
    """
    frac = np.linspace(0.0, 1.0, nbins)
    return alpha_max * frac**ramp_power


def _resample_1d(values, nbins):
    """Linearly resample an arbitrary-length 1D sequence onto `nbins`
    evenly-spaced positions (plain `np.interp`) -- decouples the LUT's own
    bin count from whatever resolution the source happened to use (a
    Blender CurveMapping evaluated at its own sample count, an Image's own
    native width, ...). A no-op reshape (still returns an array) when
    `values` already has exactly `nbins` entries. Doesn't rescale the
    VALUES themselves, only resamples their positions -- callers apply
    their own scaling (e.g. `* alpha_max`) afterward.
    """
    values = np.asarray(values, dtype=np.float64)
    if values.shape[0] == nbins:
        return values
    src_x = np.linspace(0.0, 1.0, values.shape[0])
    dst_x = np.linspace(0.0, 1.0, nbins)
    return np.interp(dst_x, src_x, values)


def _resample_rgb(rgb, nbins):
    """Same resampling as `_resample_1d`, applied per-channel to an
    `(n, 3)` or `(n, 4)` RGB(A) array. Callers here only ever use the first
    3 (RGB) channels -- alpha always comes from the alpha curve/ramp
    instead, never the color source (see module docstring's "Curve-driven
    alpha / image-driven color" section for why: exactly one mechanism
    answers "how transparent here", not two that could disagree).
    """
    rgb = np.asarray(rgb, dtype=np.float64)
    if rgb.shape[0] == nbins:
        return rgb
    src_x = np.linspace(0.0, 1.0, rgb.shape[0])
    dst_x = np.linspace(0.0, 1.0, nbins)
    out = np.empty((nbins, rgb.shape[1]))
    for c in range(rgb.shape[1]):
        out[:, c] = np.interp(dst_x, src_x, rgb[:, c])
    return out


def _build_threshold_fog_transfer_function_data(
    value_min, value_max, colormap, alpha_max, ramp_power, nbins=TF_NBINS,
    alpha_curve=None, color_rgba=None,
):
    """A CONTINUOUS ramp, not `_build_isocontour_transfer_function_data`'s
    discrete layers -- "fully transparent below a threshold, alpha ramps up
    above it" (the user's request, verbatim), enabling a genuinely foggy/
    hazy volume look rather than isocontour lines. `value_min` doubles as
    that threshold: `transfer_function.frag.glsl` already discards any
    sample outside `[tf_min, tf_max]` entirely (see its own `tex_sample < tm`
    / `> tp` checks) -- no new shader/uniform work needed, this function
    just builds a different-shaped LUT for the exact same `render_method
    ="transfer_function"` path the isocontour transfer function already
    uses. Color comes from a real yt/matplotlib colormap (`colormap`,
    sampled evenly across the bins -- registering yt's own custom colormaps
    like `"dusk"` into matplotlib is a side effect of `import yt`, done
    here rather than assumed already done); alpha is `_threshold_fog_
    alpha_ramp` (the part that's actually unit-tested). Returns the
    `(nbins, 1, 4)` uint8 RGBA array `yt_idv.opengl_support.
    TransferFunctionTexture.data` expects -- same contract as the
    isocontour builder.

    `alpha_curve`/`color_rgba`, if given, override the ramp-power/named-
    colormap mechanism for just that one channel, independently -- see
    module docstring's "Curve-driven alpha / image-driven color" section
    for where these actually come from. `alpha_curve`: any sequence of
    floats in `[0, 1]` (e.g. a Blender CurveMapping evaluated at N sample
    points), resampled to `nbins` via `_resample_1d` and then scaled by
    `alpha_max` exactly like the ramp-power shape is (`alpha_max` still
    means "the ceiling", regardless of which shape supplies the curve).
    `color_rgba`: any `(n, 3)` or `(n, 4)` sequence of RGB(A) in `[0, 1]`
    (e.g. a Blender Image treated as a 1D horizontal gradient), resampled
    the same way -- a 4th (alpha) column, if present, is IGNORED.
    """
    if color_rgba is not None:
        rgb = _resample_rgb(color_rgba, nbins)[:, :3]
    else:
        import matplotlib
        import yt  # noqa: F401 -- import side effect: registers yt's own colormaps (e.g. "dusk") into matplotlib

        cmap = matplotlib.colormaps[colormap]
        frac = np.linspace(0.0, 1.0, nbins)
        rgb = cmap(frac)[:, :3]  # (nbins, 3), [0, 1] -- drop the colormap's own alpha, we supply our own

    if alpha_curve is not None:
        alpha = _resample_1d(alpha_curve, nbins) * alpha_max
    else:
        alpha = _threshold_fog_alpha_ramp(nbins, alpha_max, ramp_power)

    rgba = np.concatenate([rgb, alpha[:, None]], axis=1)
    return np.clip(rgba * 255.0, 0, 255).astype("u1").reshape(nbins, 1, 4)


def _get_threshold_fog_transfer_function_data(
    value_min, value_max, colormap, alpha_max, ramp_power, alpha_curve=None, color_rgba=None,
):
    """Cached -- same reasoning as `_get_isocontour_transfer_function_data`,
    same "explicit values, not module constants" override mechanism.
    `alpha_curve`/`color_rgba` cross `custom_params` as plain lists (not
    numpy arrays -- see `operators._evaluate_alpha_curve`/`_sample_image_
    as_1d_rgba`), converted to tuples here for the cache key since a plain
    list isn't hashable.
    """
    alpha_key = tuple(alpha_curve) if alpha_curve is not None else None
    color_key = tuple(tuple(row) for row in color_rgba) if color_rgba is not None else None
    key = (colormap, value_min, value_max, alpha_max, ramp_power, alpha_key, color_key)
    cached = _fog_transfer_function_cache.get(key)
    if cached is not None:
        return cached
    data = _build_threshold_fog_transfer_function_data(
        value_min, value_max, colormap, alpha_max, ramp_power,
        alpha_curve=alpha_curve, color_rgba=color_rgba,
    )
    _fog_transfer_function_cache[key] = data
    return data


def _build_mip_alpha_colormap_data(
    colormap, alpha_max, ramp_power, nbins=MIP_ALPHA_NBINS, alpha_curve=None, color_rgba=None,
):
    """A `ColormapTexture.data`-shaped `(nbins, 4)` float32 RGBA array (NOT
    the `(nbins, 1, 4)` uint8 shape `TransferFunctionTexture.data` expects
    -- `ColormapTexture` is a 1D texture, matches `yt_idv.opengl_support.
    ColormapTexture._observe_colormap_name`'s own `cmap(linspace(0, 1,
    256))` output format directly) with alpha ramping from 0 at position 0
    to `alpha_max` at position `nbins - 1`, `ramp_power` shaping the curve
    exactly like `_threshold_fog_alpha_ramp` (reused directly -- same
    shape, same reasoning, different caller). Position along this array
    corresponds to where `apply_colormap_alpha.frag.glsl` samples `cm_tex`,
    i.e. the max-intensity-selected value's own position between the
    component's per-view auto-computed `cmap_min`/`cmap_max` -- so "low end"
    here means "low relative to whatever's currently visible", not a fixed
    physical density, unlike `_build_threshold_fog_transfer_function_data`.

    `alpha_curve`/`color_rgba` override the ramp-power/named-colormap
    mechanism, independently -- identical contract to
    `_build_threshold_fog_transfer_function_data`'s own parameters of the
    same name; see that function's docstring.
    """
    if color_rgba is not None:
        rgb = _resample_rgb(color_rgba, nbins)[:, :3]
    else:
        import matplotlib
        import yt  # noqa: F401 -- import side effect: registers yt's own colormaps (e.g. "dusk") into matplotlib

        cmap = matplotlib.colormaps[colormap]
        frac = np.linspace(0.0, 1.0, nbins)
        rgb = cmap(frac)[:, :3]

    if alpha_curve is not None:
        alpha = _resample_1d(alpha_curve, nbins) * alpha_max
    else:
        alpha = _threshold_fog_alpha_ramp(nbins, alpha_max, ramp_power)

    return np.concatenate([rgb, alpha[:, None]], axis=1).astype("f4")


def _get_mip_alpha_colormap_data(colormap, alpha_max, ramp_power, alpha_curve=None, color_rgba=None):
    """Cached -- same reasoning as `_get_isocontour_transfer_function_data`,
    same "explicit values, not module constants" override mechanism, same
    tuple-conversion-for-hashability treatment of `alpha_curve`/`color_rgba`
    as `_get_threshold_fog_transfer_function_data`."""
    alpha_key = tuple(alpha_curve) if alpha_curve is not None else None
    color_key = tuple(tuple(row) for row in color_rgba) if color_rgba is not None else None
    key = (colormap, alpha_max, ramp_power, alpha_key, color_key)
    cached = _mip_alpha_colormap_cache.get(key)
    if cached is not None:
        return cached
    data = _build_mip_alpha_colormap_data(
        colormap, alpha_max, ramp_power, alpha_curve=alpha_curve, color_rgba=color_rgba,
    )
    _mip_alpha_colormap_cache[key] = data
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
    called with `(component, custom_params)` right before `camera.update()`/
    `rc.run()` -- `custom_params` is `request.volume.custom_params`, the
    SAME dict `operators.build_volume_params` already merges the add-on
    Preferences' yt_idv settings into (see `_configure_transfer_function`'s
    own docstring for the full override mechanism).

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

    First calls `_get_volume_scene` (dispatches to `custom_params["yt_idv_
    volume_provider"]`, or `_default_volume_provider` -- see module
    docstring "Custom volume setup"), then `_render_signature_changed` to
    decide whether anything that would affect the image actually changed
    since the last frame THIS object was rendered -- if not, everything
    below this point (component config, camera update, depth clip, and
    critically `rc.run()` itself) is skipped entirely and the previous
    `RenderResult` is returned as-is.
    """
    height, width = request.depth.data.shape
    render_size = max(height, width)
    dataset_id = request.volume.dataset.dataset_id

    rc, scene, ds, volume_changed = _get_volume_scene(request, render_size)
    needs_render = _render_signature_changed(dataset_id, request, render_size, volume_changed)
    if not needs_render:
        cached_result = _render_result_cache.get(dataset_id)
        if cached_result is not None:
            return cached_result
        # No cached result despite an "unchanged" signature (e.g. this is
        # somehow the first-ever call for this dataset_id, or a previous
        # call raised before reaching the cache write below) -- fall
        # through and render for real rather than returning nothing.

    # Real yt_idv detail, load-bearing now that one render_context can be
    # shared by several flagged objects with DIFFERENT scenes (see
    # `_get_volume_scene`): `BaseContext.add_scene` does nothing beyond
    # `self.scene = SceneGraph.from_ds(...)`, and `OffscreenRenderingContext.
    # run()` renders `self.scene` -- NOT whatever `scene` object this
    # function happens to be holding a reference to. Whenever `_get_volume_
    # scene` reused a cached scene (no fresh `add_scene()` call this frame),
    # `rc.scene` may still point at whatever a DIFFERENT object's (or an
    # earlier frame's) `add_scene()` call last set it to. Re-asserting it
    # here, every call, costs nothing and is exactly what `add_scene` itself
    # would have set it to -- this is what makes per-object scene caching
    # correct even when several objects share one render_context.
    rc.scene = scene

    custom_params = request.volume.custom_params or {}
    for component in scene.components:
        configure_component(component, custom_params)
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

    result = RenderResult(
        rgba=np.ascontiguousarray(image),
        alpha_mode="PREMULTIPLIED",  # approximation -- see module docstring
        color_space="LINEAR",
    )
    _render_result_cache[dataset_id] = result
    return result


def _configure_max_intensity(component, custom_params):
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


def _configure_transfer_function(component, custom_params):
    """`custom_params` (from `VolumeObjectParams.custom_params`, see
    `_render_with_yt_idv`) can override any of `TF_VALUE_MIN`/`MAX`/
    `TF_N_LAYERS`/`TF_COLORMAP` via the keys `yt_idv_tf_value_min`/
    `yt_idv_tf_value_max`/`yt_idv_tf_n_layers`/`yt_idv_tf_colormap` --
    `operators.build_volume_params` populates these from the add-on
    Preferences' `yt_idv` settings (see `operators.VolumeOverridePreferences`
    /`operators._yt_idv_custom_params`), which is how the Properties panel
    UI actually reaches this module without growing `external_render.py`'s
    frozen dataclass contract or making this module `bpy`-dependent -- a
    non-Blender caller building its own `RenderRequest` by hand (e.g. the
    exploration scripts) just doesn't set these keys and gets the module
    constants below as before.
    """
    component.render_method = "transfer_function"
    value_min = custom_params.get("yt_idv_tf_value_min", TF_VALUE_MIN)
    value_max = custom_params.get("yt_idv_tf_value_max", TF_VALUE_MAX)
    n_layers = int(custom_params.get("yt_idv_tf_n_layers", TF_N_LAYERS))
    colormap = custom_params.get("yt_idv_tf_colormap", TF_COLORMAP)
    _apply_transfer_function_settings(
        component, value_min, value_max,
        _get_isocontour_transfer_function_data(value_min, value_max, n_layers, colormap),
    )


def yt_idv_max_intensity_renderer(request: RenderRequest) -> RenderResult:
    """`RENDERER_MODES["YT_IDV_MAX_INTENSITY"]` -- see module docstring for
    exactly what is and isn't real here yet.
    """
    return _render_with_yt_idv(request, _configure_max_intensity)


def _configure_max_intensity_alpha(component, custom_params):
    """`custom_params` can override `MIP_ALPHA_COLORMAP`/`MAX`/`RAMP_POWER`
    via `yt_idv_mip_alpha_colormap`/`yt_idv_mip_alpha_max`/`yt_idv_mip_
    alpha_ramp_power` -- see `_configure_transfer_function`'s docstring for
    the full override mechanism (same pattern, different keys). Also reads
    `yt_idv_alpha_curve`/`yt_idv_color_image_rgba` (shared with
    `_configure_threshold_fog` -- one curve/image setting drives both
    continuous-ramp modes) -- see `_build_mip_alpha_colormap_data`'s
    docstring for what these override and module docstring's "Curve-driven
    alpha / image-driven color" section for where they come from.
    """
    component.render_method = "max_intensity_alpha"
    colormap = custom_params.get("yt_idv_mip_alpha_colormap", MIP_ALPHA_COLORMAP)
    alpha_max = custom_params.get("yt_idv_mip_alpha_max", MIP_ALPHA_MAX)
    ramp_power = custom_params.get("yt_idv_mip_alpha_ramp_power", MIP_ALPHA_RAMP_POWER)
    alpha_curve = custom_params.get("yt_idv_alpha_curve")
    color_rgba = custom_params.get("yt_idv_color_image_rgba")
    component.colormap.data = _get_mip_alpha_colormap_data(
        colormap, alpha_max, ramp_power, alpha_curve=alpha_curve, color_rgba=color_rgba,
    )


def yt_idv_max_intensity_alpha_renderer(request: RenderRequest) -> RenderResult:
    """`RENDERER_MODES["YT_IDV_MAX_INTENSITY_ALPHA"]` -- plain
    `max_intensity` (single brightest sample per ray -- the same underlying
    `first_fragment` shader, unmodified), but with alpha genuinely driven by
    the SELECTED VALUE itself (transparent toward the low end of whatever's
    currently visible, opaque toward the high end) instead of `max_
    intensity`'s own binary "was anything found" alpha. See `apply_colormap_
    alpha.frag.glsl`'s docstring for the shader-level story and `MIP_ALPHA_
    COLORMAP`/`MAX`/`RAMP_POWER`'s own comment for how the ramp is defined.
    Everything else (camera mapping, depth-clip against Cycles' Z-buffer,
    square-render-and-crop) is exactly as in `yt_idv_max_intensity_renderer`
    -- see its and the module's docstrings.
    """
    return _render_with_yt_idv(request, _configure_max_intensity_alpha)


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


def _configure_threshold_fog(component, custom_params):
    """`custom_params` can override `TF_FOG_VALUE_MIN`/`MAX`/`COLORMAP`/
    `ALPHA_MAX`/`RAMP_POWER` via `yt_idv_fog_value_min`/`yt_idv_fog_value_
    max`/`yt_idv_fog_colormap`/`yt_idv_fog_alpha_max`/`yt_idv_fog_ramp_
    power` -- see `_configure_transfer_function`'s docstring for the full
    override mechanism (same pattern, different keys). Also reads
    `yt_idv_alpha_curve`/`yt_idv_color_image_rgba` (shared with
    `_configure_max_intensity_alpha` -- one curve/image setting drives both
    continuous-ramp modes) -- see `_build_threshold_fog_transfer_function_
    data`'s docstring for what these override and module docstring's
    "Curve-driven alpha / image-driven color" section for where they come
    from.
    """
    component.render_method = "transfer_function"
    value_min = custom_params.get("yt_idv_fog_value_min", TF_FOG_VALUE_MIN)
    value_max = custom_params.get("yt_idv_fog_value_max", TF_FOG_VALUE_MAX)
    colormap = custom_params.get("yt_idv_fog_colormap", TF_FOG_COLORMAP)
    alpha_max = custom_params.get("yt_idv_fog_alpha_max", TF_FOG_ALPHA_MAX)
    ramp_power = custom_params.get("yt_idv_fog_ramp_power", TF_FOG_RAMP_POWER)
    alpha_curve = custom_params.get("yt_idv_alpha_curve")
    color_rgba = custom_params.get("yt_idv_color_image_rgba")
    _apply_transfer_function_settings(
        component, value_min, value_max,
        _get_threshold_fog_transfer_function_data(
            value_min, value_max, colormap, alpha_max, ramp_power,
            alpha_curve=alpha_curve, color_rgba=color_rgba,
        ),
    )


def yt_idv_threshold_fog_renderer(request: RenderRequest) -> RenderResult:
    """`RENDERER_MODES["YT_IDV_THRESHOLD_FOG"]` -- a THIRD transfer function
    shape (alongside isocontour layers above): fully transparent below
    `TF_FOG_VALUE_MIN`, alpha ramping continuously up to `TF_FOG_ALPHA_MAX`
    as density climbs toward `TF_FOG_VALUE_MAX` -- a genuinely foggy/hazy
    volume look, and (per the request this was built for) a much better
    substrate for "sphere partially occluded by the volume"-style demos
    than the isocontour or MIP/projection modes: a real, continuous,
    tunable density-vs-opacity relationship instead of either narrow
    isocontour lines or (per this module's own documented finding) an
    uncontrolled colormap opacity "floor" that makes the whole domain read
    as nearly-solid past a shallow depth. Same `render_method=
    "transfer_function"` shader path as the isocontour transfer function
    (no new shader/uniform work -- see `_build_threshold_fog_transfer_
    function_data`'s docstring for why setting `tf_min` to the threshold is
    itself the whole "fully transparent below this" mechanism, for free).
    Everything else (camera mapping, depth-clip against Cycles' Z-buffer,
    square-render-and-crop) is exactly as in `yt_idv_max_intensity_renderer`
    -- see its and the module's docstrings.
    """
    return _render_with_yt_idv(request, _configure_threshold_fog)


def _configure_projection(component, custom_params):
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
