# yt_idv exploration

Retained exploration for `src/yt_volumes/yt_idv_renderer.py`
(the `YT_IDV_MAX_INTENSITY`/`YT_IDV_TRANSFER_FUNCTION` renderer modes) --
everything here was actually run, not just written; see each script's
docstring for what it confirmed and `images/` for the actual output.

## Why a separate venv

`bpy==5.2.0` (this repo's own dev/test dependency, see `pyproject.toml`)
isn't published for Python 3.12, and yt_idv's own supported range tops out
at 3.12 with a heavy compiled/OpenGL dependency stack besides -- so there is
currently no single Python that can import both `bpy` and `yt_idv` at once
in this sandbox. Real Blender's own embedded Python doesn't have this
conflict (it's not `bpy`-as-a-pip-wheel) -- see the main README's "yt_idv
integration" section for installing yt_idv there instead.

```
./setup_ytidv_venv.sh /path/to/ytidv-venv
source /path/to/ytidv-venv/bin/activate
python explore_basic_render.py
python explore_axis_order_bug.py
python explore_stub_end_to_end.py
python explore_transfer_function.py
python explore_bug3_banding.py
python explore_external_depth_clip.py
python explore_cycles_zbuffer_depth_clip.py
```

## Scripts, in the order they were actually run

1. **`explore_basic_render.py`** -- does yt_idv's offscreen MIP even work?
   Yes: `render_context("osmesa", ...)`, `add_scene(dd, "density")`
   (`render_method` already defaults to `"max_intensity"`), `rc.run()`,
   camera repositioning via `.update(...)` all work as documented.
   `images/01_*.png`, `images/02_*.png`.
2. **`explore_axis_order_bug.py`** -- testing a NON-square resolution (every
   yt_idv example/test uses a square one) surfaced two real yt_idv bugs --
   see the script's docstring and "Findings" below.
3. **`explore_stub_end_to_end.py`** -- exercises the actual add-on module
   (`yt_idv_renderer.yt_idv_max_intensity_renderer`) with a hand-built
   `RenderRequest`, confirming the whole thing works through this add-on's
   real code path, not just yt_idv's raw API. `images/03_*.png` through
   `images/09_*.png`.
4. **`explore_transfer_function.py`** -- a REAL transfer function (isocontour
   layers at fixed densities in log space, `yt.ColorTransferFunction.
   add_layers`), not the placeholder MIP colormap. The API itself was easy;
   the hard part was figuring out what numeric space `tf_min`/`tf_max`
   actually operate in (yt_idv's own `[0, 1]`-normalized value space, NOT
   raw physical density) -- see the script's docstring and `images/14_*.png`
   (first guess, nearly all black) / `images/15_*.png` (working, once the
   range was found).
5. **`explore_bug3_banding.py`** -- chases the period-4 banding artifact
   found while testing camera motion; rules out `sample_factor` (ray-march
   sample count) as the cause, confirms it's specific to non-square
   resolutions. See "yt_idv bug 3" below.
6. **`explore_external_depth_clip.py`** -- "the rays don't stop": a real,
   small yt_idv shader modification so a ray's integration can be stopped
   early at an externally-supplied depth (e.g. a solid occluder). Found and
   fixed a genuinely wrong doc comment on yt_idv's own `viewport` uniform
   along the way. See "External depth clip" below.
7. **`explore_cycles_zbuffer_depth_clip.py`** -- wires the depth clip above
   up to a REAL Cycles-shaped Z-buffer, using the flagged object's own
   bounding box as the registration between Blender world space and
   yt_idv's unitary domain (no new yt_idv Camera class needed). Drives the
   actual `yt_idv_max_intensity_renderer`/`yt_idv_transfer_function_renderer`
   functions with a hand-built `RenderRequest` whose depth data simulates a
   solid disc occluder the way Cycles' own Z pass would. See "Cycles
   Z-buffer wiring" below.
8. **`blender_scene/render_real_demo.py`** -- the full real thing, run
   against genuine Blender + genuine Cycles: a real sphere (real Cycles
   geometry, lit, denoised) centered at a corner of the flagged box with
   radius = half the box, composited against real yt_idv volume content via
   the add-on's actual `operators.render_current_frame()` two-pass
   pipeline -- not a hand-built `RenderRequest`, not a numpy composite. This
   run is what caught yt_idv bug 5 AND the two camera bugs (below). See
   "The real end-to-end demo" below.
9. **`blender_scene/render_partial_occlusion_demo.py`** -- the OPPOSITE
   compositing direction: a real sphere PARTIALLY occluded by real yt_idv
   volume content (rather than the reverse, occluding the volume, as in
   script 8). Needed a scene-geometry insight, not a code change -- see
   "Sphere partially occluded by the volume" below. Also runs with
   `VOLUME_OVERRIDE_MODE=threshold_fog` to exercise the new fog transfer
   function (see "Threshold fog transfer function" below) on the same
   geometry.
10. **`blender_scene/render_zoomed_core_demo.py`** -- a NEW real render mode,
    `YT_IDV_MAX_INTENSITY_ALPHA` (Maximum Intensity with a value-driven, not
    presence-driven, alpha -- transparent at the low end of what's visible,
    opaque at the high end), zoomed in on a real physical scale (a 10 kpc
    marker sphere at the galaxy's own core). See "Maximum Intensity with a
    value-driven alpha" below -- this is the one worth reading if you only
    read one section.
11. **`blender_scene/custom_volume_provider_demo.py`** (+ its sibling
    `_custom_volume_provider_example.py`) -- verifies the NEW
    `yt_idv_volume_provider` harness (replace the hardcoded IsolatedGalaxy
    sample with your own real yt setup, called once per object per frame,
    stateful) against a real render_context: identical output to the
    default provider, correct stateful reuse across calls, the skip-render
    cache actually skipping `rc.run()` when nothing changed (and NOT
    skipping it on a real camera change), and correct behavior when two
    objects share one render_context. See "Custom volume setup" below.
12. **`blender_scene/render_headless.py`** -- opens a real, externally-
    authored `.blend` file (built in any Blender, yt_idv not required
    there) and renders it through the real add-on operator and real
    yt_idv, entirely via `uv run --group ytidv` -- no separate Blender
    binary, no `--background` flag. Verified against a real 1920x1080,
    250-frame, Blender-5.3-saved file with an animated camera orbiting a
    cube. Two real findings from that run, documented in its own module
    docstring: `addon_enable` needs nothing but `sys.path` (no stub-prefs
    monkeypatch required just to go headless), and per-frame output
    filenames must be set explicitly inside the loop -- they are NOT
    auto-numbered the way a plain single-image render's are.

Camera motion + reference AMR grid-box wireframes + a marker sphere, all
driven from a real Blender scene, lives in `blender_scene/` (separate
subprocess-bridged setup, see that directory's own README).

## Findings

**yt_idv bug 1 -- wrong axis order for non-square renders (both backends,
data itself fine).** `yt_idv.scene_graph.SceneGraph.image` reads back via
`glReadPixels(0, 0, width, height, GL_RGBA, GL_FLOAT)`, which comes back
shaped `(width, height, 4)`, then calls `arr.swapaxes(0, 1)` **without
reassigning the result** -- `ndarray.swapaxes` returns a new view rather
than mutating in place, so that line is a complete no-op. Every non-square
render comes back axis-swapped relative to the conventional
`(height, width, 4)` shape. Invisible in yt_idv's own examples/tests because
they all use square dimensions. Worked around entirely on this add-on's
side (`yt_idv_renderer._fix_yt_idv_axis_order`) -- no yt_idv edit needed.

**yt_idv bug 2 -- OSMesa backend corrupts non-square renders (data itself,
not just shape).** Visibly tiled/repeated garbage
(`images/06_bug2_osmesa_corrupted_nonsquare.png`), not a clean transpose of
a correct image -- rules out "it's just bug 1 again". The obvious suspect
(`OSMesaMakeCurrent`'s `width`/`height` arguments look swapped relative to
the real OSMesa C API signature) was tried via a monkeypatched subclass and
did **not** fix it (`images/07_*.png` looks essentially the same as the
unfixed version) -- the actual cause is still open. **EGL does not have
this problem** -- only needs bug 1's axis-order fix. This is why
`yt_idv_renderer.DEFAULT_BACKEND` is `"egl"`, not `"osmesa"`, despite EGL
needing a working GPU + driver stack that OSMesa doesn't. Worth an upstream
yt_idv bug report; not investigated further here (would mean digging into
Mesa's software rasterizer internals, out of scope for an exploratory
stub).

**yt_idv bug 3 -- exact period-4 banding on non-square renders, EGL, even
after bug 1's fix.** Found while testing camera motion (see
`blender_scene/`). A raw scanline dump showed an EXACT period-4 pattern --
`[value, 0, 0, value]` repeating -- not random noise, not proportional to
scene content. Tested and REJECTED `sample_factor` (up to 10x -- see
`component_shaders.block_rendering`'s `sample_factor` trait) as the cause:
pixel-identical output regardless of its value, ruling out a ray-marching
under-sampling explanation someone might otherwise reach for. Confirmed the
SAME camera/dataset at a SQUARE resolution shows a clean, continuous
scanline -- only the aspect ratio changed between the banded and clean
case. Root cause still open (candidates: EGL surface/multisample config, a
width/height mixup in yt_idv's own viewport or grid-expansion geometry
shader specifically for non-square viewports) -- being investigated
independently. **Worked around in `yt_idv_renderer.py`: render square,
center-crop to the requested shape afterward** (see
`yt_idv_max_intensity_renderer`'s docstring) -- confirmed this produces a
clean image at the same extreme aspect ratio that showed banding before
(`images/09_extreme_aspect_via_real_module_fixed.png`, re-generated after
the fix).

**yt_idv bug 4 -- the `viewport` uniform's own doc comment is wrong.**
Found while building the external depth clip (below). `known_uniforms.
inc.glsl` documents `viewport` as `(offset_x, offset_y, 1 / screen_x,
1 / screen_y)`, implying you multiply by `.zw` for a normalized screen UV.
`yt_idv.cameras.base_camera.BaseCamera._set_uniforms` actually sets it to
the RAW `GL.glGetIntegerv(GL_VIEWPORT)` -- `(x, y, width, height)` in
pixels. You must divide, not multiply. Not fixed in yt_idv itself (it's a
stale comment, not our shader code); worked around by dividing in
`ray_tracing.frag.glsl`'s own new code, with a comment explaining why.

**Transfer function `tf_min`/`tf_max` operate in yt_idv's own `[0, 1]`
min-max-normalized value space, not raw physical field units** -- see
`explore_transfer_function.py` and `yt_idv_renderer.py`'s "Transfer
function" docstring section for the full account (confirmed by reading
`yt_idv.scene_data.block_collection.BlockCollection._load_textures`
directly, then cross-checked against `max_intensity`'s own auto-printed
`cmap_min`/`cmap_max`).

**External depth clip -- a real, small yt_idv shader modification, not a
workaround.** yt_idv's ray march had no way to stop early at an
externally-supplied depth (needed for a solid occluder to actually occlude
volume content behind it -- the flat alpha-over composite in
`blender_scene/` can only ever produce an all-or-nothing cutout, since MIP
has no per-pixel depth of its own to composite against). Added a new
uniform pair (`external_depth_tex`, `use_external_depth_clip`) to
`ray_tracing.frag.glsl` + `known_uniforms.inc.glsl` + `blocks.py` -- see
`yt_idv_external_depth_clip.patch` for the exact diff and
`explore_external_depth_clip.py` for the full verification story, including
bug 4 above (found while building this) and a Python-side gotcha
(`Texture2D`'s internal axis swap needs a specific `depth[::-1, :].T`
transform to sample where you'd expect -- now baked into
`yt_idv_renderer.set_external_depth_clip`, callers don't need to think about
it). The payoff: a real, per-pixel, sphere-shaped depth clip produces a
genuine partial "bite" taken out of the rendered nebula -- gas outside the
sphere's silhouette renders completely normally, only the overlapping
portion disappears (`images/19_final_baseline.png` vs
`images/20_final_sphere_clipped.png`). Off by default, confirmed zero
regression to existing behavior when unused (`np.allclose` against the
unclipped baseline in every test).

**Cycles Z-buffer wiring -- the external depth clip above, now driven by a
real Cycles-shaped Z-buffer, not just a same-yt_idv-session synthetic
occluder.** The flagged object's own `bounding_box_local`/`matrix_world`
(already part of `VolumeObjectParams`, no new field needed) is the
registration between Blender world space and yt_idv's unitary domain: a
unitary point `u` lives at object-local point `bbox_min + u*bbox_extent`,
then `matrix_world` into Blender world space. This ONE registration now
drives BOTH the camera (`_registered_camera_kwargs` -- position and
orientation genuinely transformed through it, replacing the old "orbit at a
fixed heuristic distance, orientation only" approach entirely) and the
depth clip (`_cycles_depth_to_yt_idv_window_depth` -- Cycles' own Z-buffer
reconstructed into world-space hit points via the exact ray math already
verified in `sample_renderers._persp_ortho_rays`, transformed through the
SAME registration into unitary space, then projected through yt_idv's own
now-correctly-positioned camera). No new yt_idv Camera subclass needed --
`BaseCamera.update(position=..., focus=..., up=..., ...)` was already
general enough, it just needed genuinely-transformed values instead of an
orbit heuristic. `explore_cycles_zbuffer_depth_clip.py` drives the REAL
`yt_idv_max_intensity_renderer`/`yt_idv_transfer_function_renderer`
functions (not a hand-rolled equivalent) with a hand-built `RenderRequest`
whose depth data simulates a solid disc occluder the way Cycles' own Z pass
would: the result (`images/22_*.png`) shows a genuine, clean DISC-shaped
hole cut exactly where the fabricated occluder sits. The math itself is
separately verified against a REAL bpy-computed `projection_matrix` in
`tests/test_yt_idv_renderer.py`, not just this end-to-end script. Still an
approximation in two specific ways, not glossed over: (1) yt_idv renders
SQUARE while Blender's camera essentially never is, so the depth
reconstruction (on Blender's real, non-square grid) gets center-padded onto
the square canvas the same way the final image gets center-cropped back
off, but a square-aspect yt_idv camera can't reproduce Blender's true
horizontal FOV; (2) the object's bounding box has to actually correspond to
where you want the dataset positioned -- a registration you place by
sizing/moving the flagged object, not inferred from the dataset itself.

**A THIRD render mode, `"projection"` (unweighted line-of-sight
integration, additive not max), added after `images/22_*.png` looked
noisy/speckled.** `yt_idv_projection_renderer` -- same registration, same
depth clip, just a different `component.render_method`. Same scene, same
depth clip, noticeably smoother/more coherent fill
(`images/23_cycles_zbuffer_disc_occlusion_projection.png`) -- though the
disc clip boundary itself still shows some jaggedness in both, which looks
like an AMR block-boundary artifact (different refinement levels
contributing their own edge near the clip boundary) rather than a
MIP-vs-projection difference; not chased further here.

**yt_idv bug 5 -- the EGL context requests no alpha bits at all.** Found
while verifying the real end-to-end demo's background transparency (below),
not while poking at internals -- the first real composited image looked
wrong (an opaque black background instead of a transparent one) and tracing
it back landed on `yt_idv/rendering_contexts/egl_context.py`'s
`config_attribs`: `EGL_RED_SIZE`/`EGL_GREEN_SIZE`/`EGL_BLUE_SIZE` = 8 each,
but no `EGL_ALPHA_SIZE` entry at all. With no alpha bits in the surface's
framebuffer, `scene_graph.SceneGraph.image`'s `glReadPixels(..., GL_RGBA,
...)` fallback (used for every top-level offscreen render this add-on does)
read back alpha exactly `1.0` for literally every pixel, including
genuinely-empty background ones (confirmed: RGB correctly `(0, 0, 0)` there,
alpha still `1.0`). Fixed by adding `EGL_ALPHA_SIZE, 8` to the config
request -- see `yt_idv_external_depth_clip.patch` (now covers this fix too,
alongside the external-depth-clip changes) for the exact diff. Confirmed
fixed by re-running the same real-demo pipeline before/after: background
corner-pixel alpha went from `1.0` to `0.0`.

**The real end-to-end demo -- everything above, run against genuine
Blender + genuine Cycles, not a hand-built `RenderRequest`.**
`blender_scene/render_real_demo.py` builds a real scene (a 2x2x2 "flagged
volume box" cube, a sphere centered at one of its corners with radius = half
the box's edge length -- i.e. it genuinely pokes into the box rather than
either missing it or swallowing it whole, since the corner-to-center
distance `sqrt(3) ≈ 1.73` exceeds the sphere's radius of `1`), a real
Cycles camera and sun lamp, `film_transparent = True`, then runs the add-on's
actual `operators.render_current_frame()` -- real Pass 1 Cycles beauty+Z
render, real `read_depth_buffer`, real per-object `RenderRequest`
construction, real Pass 2 compositor Alpha-Over graph and final render --
completely unmodified, monkey-patching only `RENDERER_MODES`'s yt_idv entry
to bridge to the throwaway yt_idv venv via `ytidv_worker.py`'s new
`render_with_depth` subcommand (real bpy/yt_idv still can't coexist in one
Python here -- see "Why a separate venv" above; this substitution is the
only thing not exactly what a real Blender install would do in-process).
Result (`images/24_real_demo_sphere_and_volume.png`): the sphere renders as
real lit Cycles geometry, the volume fills the rest of the box's silhouette,
and a genuine crescent-shaped gap is visible at the sphere's base where the
real depth clip stops the volume's ray march at the sphere's own Cycles
Z-buffer -- not a flat all-or-nothing cutout. Background transparency is
verified two ways, not just asserted: a direct alpha-value check on a
background pixel (`~0.0`, logged by the script), and a second export
(`images/25_real_demo_over_checkerboard.png`) that composites the same
result over a checkerboard -- the checker pattern shows through everywhere
neither the sphere nor the volume is present, with no black rectangle
anywhere, which a genuinely-opaque background could not produce.

**Sphere partially occluded by the volume -- the OPPOSITE compositing
direction, requested next.** Every demo above shows a solid sphere occluding
the volume (via the external depth clip). The reverse direction -- the
volume drawing OVER part of a solid sphere -- needed NO code change at all:
`compositor_build.build_or_update_graph`'s Alpha-Over chain already draws
the volume layer over the Cycles beauty pass unconditionally, using the
volume's own per-pixel alpha; the depth clip's only job is preventing the
volume from ALSO drawing content that's actually behind the sphere. So this
was purely a scene-geometry problem: get a sphere partly behind real,
opaque-looking volume content, from the camera's point of view.

First attempt got this wrong in an instructive way: center the sphere near
the dataset's own density peak (`IsolatedGalaxy`'s peak sits at unitary
(0.504, 0.500, 0.500) -- essentially the domain's own center, confirmed via
`ds.all_data().quantities.max_location("density")`, not a guess), reasoning
that the bright peak alone would occlude just the nearby part of the
sphere. Result: the volume hid the ENTIRE sphere, everywhere it was tested.
This connects back to an already-documented finding just below (the current
colormap gives even very LOW densities a nonzero opacity "floor", not true
transparency) -- it means the WHOLE box interior reads as close to fully
opaque once a ray has traveled even a modest distance into the domain, not
just near the peak, so any sphere placed meaningfully inside the box's
volume gets hidden completely regardless of where the peak is.

The actual fix: since Cycles never renders the box at all (it's hidden,
flagged for override), the ONLY thing determining whether the volume covers
a given sphere pixel is whether that same 2D screen pixel has meaningful
alpha in the volume's own render -- nothing to do with real 3D "inside the
box." `blender_scene/render_partial_occlusion_demo.py` positions the sphere
BEHIND the box relative to the camera (past its far boundary along the
view direction) and OFFSET SIDEWAYS, so its silhouette partially overlaps
the box's screen footprint (hidden under real volume color) while the rest
sticks out past the box's silhouette on one side (fully visible, genuinely
zero volume alpha there) -- tuned using `bpy_extras.object_utils.
world_to_camera_view` to check the sphere's projected screen columns
against the box's (box: ~111-407 px at 512x512 for this camera; sphere
tuned to ~356-473 px, a real ~51px overlap and ~66px clear peek-out).
Verified directly against real pixel data, not just eyeballing the image:
the script samples one point on the sphere expected to be occluded and one
expected to be visible, and prints both the real Cycles-only color and the
final composited color at each. Confirmed: the "behind the box" sample's
final color differs substantially from its own beauty-only color (the
volume genuinely drew over it); the "peeking out" sample's final color
matches its beauty-only color EXACTLY (`|final - beauty| == 0.0000`) --
that part of the sphere is untouched. Result:
`images/32_partial_occlusion_sphere_and_volume.png` -- the sphere's real,
lit reddish-brown material is clearly visible on one side and clearly
replaced by the volume's own bright/hazy color on the other, a genuine
partial occlusion, not a subtle tint. Background transparency re-verified
the same two ways as the other demo (`images/33_partial_occlusion_over_
checkerboard.png`).

**Threshold fog transfer function -- a fourth render mode, requested
directly: "a colormap that is fully transparent except at some threshold
value... ramp up the alpha to make it look foggy."** No new shader/uniform
work needed: `transfer_function.frag.glsl` already discards any sample
below `tf_min` entirely, so setting `tf_min` to the threshold gives "fully
transparent below this" for free -- `_build_threshold_fog_transfer_
function_data` just builds a CONTINUOUS alpha ramp (rising from 0 at the
threshold to `TF_FOG_ALPHA_MAX`) instead of the isocontour mode's discrete
layers, reusing the exact same `render_method="transfer_function"` shader
path. Two real things surfaced while building this on the partial-occlusion
demo's real Cycles+depth-clip scene, rather than a synthetic test:

1. **Another real yt_idv bug**: the first real render's background
   composited as opaque black instead of transparent -- NOT a repeat of
   bug 5 (already fixed), traced to `passthrough.frag.glsl` (used as
   `block_rendering.transfer_function`'s own SECOND pass) unconditionally
   setting `color.a = 1.0`, discarding the first pass's real per-pixel
   accumulated alpha entirely. This shader is shared by several
   genuinely-always-opaque usages (outlines, text, line plots -- see
   `shaderlist.yaml`'s other `second_fragment: passthrough` entries), so it
   couldn't just be edited in place; fixed with a new sibling shader,
   `passthrough_alpha.frag.glsl` (identical, minus that one line), used
   ONLY for `block_rendering.transfer_function`'s `second_fragment` (both
   geometry variants). This also fixes the PRE-EXISTING isocontour
   `YT_IDV_TRANSFER_FUNCTION` mode's background-alpha correctness, never
   specifically checked before (`render_real_demo.py`'s own background
   verification had only ever run against `max_intensity`/`projection`).
2. **A real re-tuning need, not a bug**: even after that fix, the fog was
   invisible -- the pre-existing isocontour `TF_VALUE_MIN`/`MAX` (1e-6 to
   2e-3) produced essentially zero accumulated opacity against this
   REGISTERED camera. Root cause: the earlier camera-matrix bug fix (see
   "Two real bugs in THIS ADD-ON's own code" below) correctly changed the
   actual geometry every yt_idv render uses, which also shifts which
   density values a given camera/scene combination samples along its rays
   -- `transfer_function.frag.glsl`'s accumulation is a REAL, physically-
   based front-to-back opacity integral (sensitive to that), unlike
   `max_intensity`/`projection`'s effectively-binary "was anything found"
   check via their own auto-computed per-view colormap bounds. Confirmed by
   sweeping `value_min`/`value_max`/`ramp_power` against a real cached
   Cycles+depth-clip request (not guessing): a front-loaded ramp
   (`TF_FOG_RAMP_POWER < 1.0`, alpha rises fast just past the threshold
   rather than climbing gradually to `TF_FOG_VALUE_MAX`) raised the peak
   accumulated alpha from ~0.16 to ~0.60 on the same exact scene, because it
   spreads high per-step alpha over a WIDER slice of the density values
   actually encountered instead of concentrating it near the very top of
   the range. The pre-existing isocontour constants are NOT re-tuned here
   (see main README "What's next") -- this fog mode's own constants are.

Result: `images/34_partial_occlusion_fog_sphere_and_volume.png` -- a soft,
glowing haze with a brighter core, genuinely translucent (confirmed against
a checkerboard, `images/35_partial_occlusion_fog_over_checkerboard.png`)
rather than MIP/projection's harder, near-uniform floor fill -- run via the
SAME `render_partial_occlusion_demo.py` script, `VOLUME_OVERRIDE_MODE=
threshold_fog`.

**Maximum Intensity with a value-driven alpha -- a fifth render mode,
`YT_IDV_MAX_INTENSITY_ALPHA`, requested directly: zoom in on a real 10 kpc
scale and see MIP with transparency at the low end, opacity at the high
end.** Plain `max_intensity`'s own alpha turns out to be a binary "was
anything found along this ray" flag, completely decoupled from the selected
value (`apply_colormap.frag.glsl` always overwrites `color.a` with that
flag) -- so it CAN'T show a transparent-low/opaque-high gradient no matter
how the colormap is built. Fixed with a new sibling shader,
`apply_colormap_alpha.frag.glsl` (identical, but keeps the colormap
texture's own alpha channel instead of overriding it), used by a new
`max_intensity_alpha` `block_rendering` combo -- `_build_mip_alpha_
colormap_data` in `yt_idv_renderer.py` builds a real colormap with a
value-driven alpha ramp (reusing the same `_threshold_fog_alpha_ramp` shape
as the fog transfer function above), assigned directly to `component.
colormap.data`. Rides on MIP's EXISTING per-view auto `cmap_min`/`cmap_max`
rather than needing its own fixed dataset-tuned bounds, so (unlike the fog
mode) it didn't need dataset-specific recalibration.

Zoom distance took real, empirical work, and the story is worth reading in
full in `render_zoomed_core_demo.py`'s own module docstring: the naive
instinct (camera a few multiples of the marker sphere's own 10 kpc away)
produces a bizarre, sharp TRIANGULAR silhouette instead of organic gas
structure -- at that distance the camera is pressed up against (or inside)
a SINGLE AMR block's own triangulated bounding cube, so what's on screen is
that block's flat geometric face, not resolved density. Backing off to
300-500 kpc (30-50x farther than that instinct) lets the frame span many
nested blocks and the structure resolves cleanly -- confirmed by comparing
renders across 15/30/50/300/400/500/650 kpc. The marker sphere is still
there at 500 kpc, genuinely tiny (confirmed directly against the real
Cycles beauty pass) -- itself an honest answer to "zoom in on a 10 kpc
sphere" at this dataset's actual resolution: the length scale where the
surrounding structure resolves and the length scale of a 10 kpc marker
don't cooperate. `MIP_ALPHA_RAMP_POWER`/`COLORMAP` were swept (5 colormaps
x 4 ramp powers) against this same real scene's cached depth data, no
Cycles re-render per candidate. Result:
`images/36_zoomed_core_sphere_and_volume.png` -- a soft, glowing core (dusk
colormap: blue outer, orange/white core) with the tiny marker sphere
visible right at the peak, genuinely transparent at the edges (confirmed
against a checkerboard, `images/37_*`).

**Follow-up -- "I can't quite see it."** The first version's 10 kpc yellow
sphere was genuinely hard to spot, for three distinct, real reasons, all
fixed in `render_zoomed_core_demo.py`'s `build_scene()`: (1) yellow sat too
close to "dusk"'s own warm orange/white core color, so the sphere blended
in rather than contrasted -- switched to a saturated green; (2)
`MIP_ALPHA_MAX` (1.0) let the volume's own near-camera material build up to
FULL opacity in front of the sphere, hiding it completely via Alpha-Over
regardless of its own real depth-clip visibility -- lowered to 0.7, a real,
general fix to `yt_idv_renderer.py`'s own default, not a one-off for this
demo; (3) the sphere itself was also enlarged, 10 -> 35 kpc. Re-rendered
`images/36_*`/`37_*` (same paths, overwritten) show the green marker
reading clearly against the core.

**Visual character of the render itself (not a bug, just what MIP with the
current default colormap looks like):** the domain box's silhouette,
mostly filled with a dim, near-uniform dark-red tone, plus a bright
blue/green/white blob at the galaxy's actual density peak. The dim fill
across the WHOLE box (not just near the blob) suggests the current colormap
gives very low densities a nonzero floor color/alpha rather than true
transparency -- this is exactly what led to building a real transfer
function (above).

## What's NOT covered here

Blender-side integration (flagged object + camera driving the actual
two-pass pipeline, reference grid-box wireframes for the dataset's AMR
structure, camera-motion-over-frames, a composited marker sphere) lives in
`scripts/yt_idv_exploration/blender_scene/` -- see that directory's own
README for why it's a separate subprocess-bridged setup (bpy and yt_idv
can't share one Python here either, same root cause as above).
`render_frames_in_blender.py` (camera-motion-over-frames, grid-box
wireframes, the original marker sphere) predates the Cycles Z-buffer wiring
and still uses the OLD flat alpha-over composite technique --
`render_real_demo.py` (above) is the real-depth-clip redo, but only for a
single still frame with a fixed sphere/camera, not the full multi-frame
orbit `render_frames_in_blender.py` covers; re-doing THAT one with the real
depth clip across an animated camera path is still open.

The add-on's real yt_idv Preferences-panel settings (`operators.
YtIdvRenderSettings` -- see main README "Custom properties ->
VolumeObjectParams.custom_params" -> "yt_idv-specific parameters") also
aren't exercised by any script here: `_bridge_render`/`ytidv_worker.py`'s
`_build_request`/`_build_request_full` always hand-build a `VolumeObjectParams`
with `custom_params={}`, since these scripts construct their own
`RenderRequest` directly rather than going through `operators.
build_volume_params`. Real add-on usage (a real Blender with yt_idv
installed, no bridge, `render_fn(request)` called directly) is unaffected
-- this is a limitation of this sandbox's own dev-only bridge scripts, not
of the feature itself, which was verified directly against real Blender
(see SUMMARY.md).
