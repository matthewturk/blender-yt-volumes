# Session summary

Blender add-on: objects flagged with a custom property render via an
external volume renderer instead of Cycles, depth-clipped against Cycles'
Z pass, composited back in via premultiplied Alpha Over. Built for
eventual integration with scientific datasets (yt-loadable simulations,
not just VDB). Full design writeup is in `README.md` -- this file is just
a map of what happened this session and what's still open.

**Status**: 66/66 tests pass (`uv run pytest`, or `env -u LD_PRELOAD uv run
pytest` if your shell's `LD_PRELOAD` references something missing). Add-on
is installable and runnable in real Blender, on both classic (3.x/4.x) and
5.0+ compositor APIs now (see "Known gaps" #1 below). Confirmed actually
running end to end against a real Blender 5.x install (not just the pip
test module) -- see "What happened when you actually ran it" below for the
three real bugs that surfaced and got fixed. `render_volume()` itself is
still a stub, EXCEPT for a new exploratory renderer mode that calls the
real yt_idv volume renderer -- see "yt_idv integration" below.

## What exists

| File | What it does |
|---|---|
| `src/yt_volumes/__init__.py` | `bl_info`, register/unregister |
| `properties.py` | Object flag + shading knobs + `dataset_path`/`dataset_kind` + time-mapping fields, UI panel |
| `external_render.py` | **The frozen contract** -- all dataclasses (`CameraParams`, `DepthBuffer`, `VolumeObjectParams`, `DatasetParams`, `VersionInfo`, `RenderRequest`/`RenderResult`) + stubbed `render_volume()` |
| `scene_sync.py` | Hidden sync scene, main-scene visibility toggle for flagged objects |
| `change_tracking.py` | Content-hash "did this change since last call" tracking |
| `compositor_build.py` | Programmatic compositor graphs (Cycles+Z capture pass, then Image-node Alpha Over chain), both classic and 5.0+ APIs |
| `operators.py` | The two-pass-per-frame render orchestration, `RenderRequest` building, F12 takeover, verbose step-by-step logging, Render Result display |
| `handlers.py` | `render_cancel` safety net, keymap install/uninstall |
| `sample_renderers.py` | Worked example: real rays + real depth-clip, trivial `depth * constant` shading -- not the real renderer |
| `yt_idv_renderer.py` | Exploratory: calls the REAL yt_idv volume renderer (offscreen OpenGL, Maximum Intensity Projection), hardcoded IsolatedGalaxy sample dataset -- see "yt_idv integration" below |

Plus `scripts/verify_*.py` (empirical checks, see below), `scripts/
yt_idv_exploration/` (yt_idv exploration, findings, and a Blender-scene
camera-motion test, all retained/committed), and a full `tests/` suite.

## What happened when you actually ran it (real Blender 5.x)

Three real, distinct bugs surfaced, all now fixed and covered by tests:

1. **Depth read-back crashed**: `Volume override render failed: index 0 is
   out of bounds for axis 2 with size 0`. Root cause: `compositor_build.py`
   wrote the pass-1 beauty/depth EXRs with a *named* `file_output_items`
   entry (Blender 5.0+'s File Output node). Blender writes that as a named
   EXR "part" -- and `bpy.data.images.load()` (what `read_depth_buffer()`
   uses) doesn't reliably decode a named-part EXR: it silently returns a
   0×0/0-channel image instead of raising, which is exactly the "broken EXR
   decoder" previously (and wrongly) blamed on the pip `bpy==5.2.0` test
   wheel specifically -- see "What's been empirically verified" below, this
   turned out to be the same bug, not a packaging defect. **Fixed** by
   giving those items an empty name (`_configure_single_file_output` in
   `compositor_build.py`); `read_depth_buffer()` also now fails loudly with
   the actual on-disk facts (path/size/channels) if this class of bug ever
   recurs, instead of a bare `IndexError`.
2. **No visible result after F12**: EXR files were written correctly, but
   nothing showed up in the Image Editor / Render Result window. Root
   cause: the pipeline's internal `bpy.ops.render.render()` calls run
   `EXEC_DEFAULT` (no popup) so pass 2 can't race the engine-swap around it
   -- unlike a normal F12 press, which uses `INVOKE_DEFAULT` and pops the
   render window itself. **Fixed** by calling `bpy.ops.render.view_show()`
   (Blender's own "just display whatever's already rendered" operator, safe
   to call after the fact) once the pipeline finishes -- see
   `operators._show_render_result()`.
3. **A slow pass 1 render looked like a hang** with no visibility into
   which step was actually running. **Fixed** by adding timestamped,
   per-step logging through the whole two-pass pipeline (both render calls,
   the depth read-back, each object's `render_volume()` call, each EXR
   write) -- gated on the new `VolumeOverridePreferences.verbose_logging`
   (default on), printed to Blender's system console. See `operators._log`/
   `_StepTimer`.

Also, per your request: the debug/sample renderers' placeholder color
default (`VolumeOverrideProperties.color`, and the matching fallbacks in
`external_render.debug_fake_render_volume`/
`sample_renderers.constant_density_renderer`) is now a loud, obviously-fake
**red** (`1.0, 0.0, 0.0`) instead of a plausible-looking pale blue, so
placeholder output is unmistakable against Cycles' own render while wiring
things up. Change it in the object's Volume Override panel once you're past
that stage.

## What's been empirically verified (not assumed)

Run under pip-installable `bpy==5.2.0` (headless, no GUI) since real
Blender wasn't available at the *start* of this session -- some of these
were later independently confirmed against a real Blender 5.x install too
(noted below):

- **Cycles' Z pass is PLANAR** (view-space depth), not Euclidean ray
  distance as commonly assumed. No-hit sentinel is exactly `1e10`.
  (`scripts/verify_zpass.py`)
- **`Object.calc_matrix_camera()` + `matrix_world` round-trips to real
  Cycles pixels** for both PERSP and ORTHO, sub-pixel accurate.
  (`scripts/verify_camera_projection.py`)
- **Equirectangular panorama mapping**, camera-local: `u = atan2(x,-z)/(2π)+0.5`,
  `v_from_top = 0.5 - asin(y/r)/π`. Confirmed with real marker renders.
  (`scripts/verify_pano_equirect.py`)
- **Row order is `BOTTOM_LEFT_ORIGIN`, and this is now closed-loop
  confirmed**, not just assumed. Previously this was only checkable via a
  raw-on-disk-EXR fallback reader because the add-on's actual path
  (`bpy.data.images.load().pixels`) reproducibly returned a 0×0 image --
  misdiagnosed at the time as a defect specific to the pip test wheel. It
  wasn't: it was bug #1 above (named EXR parts), independently confirmed on
  a real Blender 5.x install via the actual add-on. Once fixed,
  `bpy.data.images.load()` succeeds and directly confirms
  `BOTTOM_LEFT_ORIGIN`, matching `operators.DEPTH_ROW_ORDER`'s assumption.
  All three `scripts/verify_*.py` scripts now report
  `readback_method: "bpy.data.images.load"` (no fallback) after the fix.

## yt_idv integration (new this session)

Wrote `yt_idv_renderer.py` -- three `RENDERER_MODES` entries,
`YT_IDV_MAX_INTENSITY`, `YT_IDV_TRANSFER_FUNCTION`, and `YT_IDV_PROJECTION`,
all calling the REAL yt_idv volume renderer (you placed a checkout at
`yt_idv/`) instead of a toy stand-in. Deliberately scoped down per your
brief: hardcoded `IsolatedGalaxy` sample dataset (no dataset-selection
wiring). Full detail in `yt_idv_renderer.py`'s own module docstring -- this
is a summary, not a replacement for it.

**Verified for real, not just written**: no single Python here can import
both `bpy` (pinned to 5.2.0/3.13 for this repo's own dev venv) and `yt_idv`
(tops out at 3.12) at once, so verification happened in a throwaway 3.12
venv (`scripts/yt_idv_exploration/setup_ytidv_venv.sh`) -- real dataset
load, real offscreen OpenGL render, real camera motion driven by
`operators.build_camera_params()` across multiple frames via a subprocess
bridge, real images inspected, not just "it imported without error." All of
it retained in `scripts/yt_idv_exploration/` (scripts + reference images +
findings, not thrown away after the session).

**Real transfer function** (your ask: "isocontours at fixed densities in
log space"): `yt.ColorTransferFunction.add_layers`, not a placeholder
colormap. The API itself was easy; finding the right numeric range for
`tf_min`/`tf_max` took real digging -- they operate in yt_idv's own
`[0, 1]`-normalized value space (confirmed by reading `BlockCollection.
_load_textures` directly), not raw physical density, and a first guess
based on that space's nominal bounds came out nearly all black
(`scripts/yt_idv_exploration/images/14_*.png`) before cross-checking
against MIP's own auto-printed per-view max range found the actual useful
window (`images/15_*.png`).

**External depth clip -- a real, small yt_idv shader modification, not a
workaround**: yt_idv's ray march had no way to stop early at an
externally-supplied depth, so a solid occluder could never do more than an
all-or-nothing flat cutout. Added a new uniform pair to `ray_tracing.
frag.glsl`/`known_uniforms.inc.glsl`/`blocks.py` (patch retained at
`scripts/yt_idv_exploration/yt_idv_external_depth_clip.patch`), off by
default, zero effect on existing behavior when unused. First verified with
a real per-pixel, sphere-shaped depth (analytic ray/sphere intersection): a
genuine partial "bite" taken out of the rendered nebula, gas outside the
sphere's silhouette rendering completely normally -- the actual "sphere
partly in front of it" result you asked for, not achievable with a flat
alpha-over composite (MIP has no per-pixel depth of its own to composite
against). Exposed as `yt_idv_renderer.set_external_depth_clip`.

**Then wired to Blender's real Cycles Z-buffer** (your idea: "just use a
transform to unitary space from the cycles renderer" -- no new yt_idv
Camera class needed, confirmed): the flagged object's own
`bounding_box_local`/`matrix_world` (already part of `VolumeObjectParams`)
is the registration between Blender world space and yt_idv's unitary
domain. This ONE registration now drives BOTH the camera
(`_registered_camera_kwargs` -- real position AND orientation, replacing
the old orbit-heuristic entirely) and the depth clip
(`_cycles_depth_to_yt_idv_window_depth` -- Cycles' Z-buffer reconstructed
into world points via the exact ray math already verified in
`sample_renderers._persp_ortho_rays`, transformed through the registration,
then projected through yt_idv's own now-correctly-positioned camera).
Verified two ways: a fabricated disc-shaped Cycles Z-buffer produces a
genuine disc-shaped hole in the render end to end
(`scripts/yt_idv_exploration/explore_cycles_zbuffer_depth_clip.py`), and
the underlying math is separately verified against a REAL bpy-computed
`projection_matrix` in `tests/test_yt_idv_renderer.py` (not hand-rolled
numbers). Still honestly approximate: yt_idv renders square while
Blender's camera essentially never is, and the object's bounding box has to
actually correspond to where you want the dataset positioned.

Also added a THIRD render mode, `YT_IDV_PROJECTION` (unweighted
line-of-sight integration, additive not max) -- you flagged the first
disc-occlusion render as noisy/speckled; projection mode on the exact same
scene looks noticeably smoother. The disc clip boundary itself still shows
some jaggedness in both, which looks like an AMR block-boundary artifact
rather than a MIP-vs-projection difference -- not chased further.

**Five real yt_idv bugs found and worked around/fixed** (three from
non-square resolutions -- every yt_idv example/test uses a square one,
Cycles renders essentially never are -- one from building the depth clip,
one from the real end-to-end demo below) -- see README "yt_idv integration"
and `yt_idv_renderer.py`'s module docstring for full detail:
1. Non-square renders come back axis-swapped (`(width, height, 4)` instead
   of `(height, width, 4)`) -- a real no-op bug in yt_idv's own
   `SceneGraph.image` (`arr.swapaxes(0, 1)` without reassigning). Fixed on
   this side, no yt_idv edit.
2. OSMesa backend genuinely CORRUPTS (not just mis-shapes) non-square
   renders -- tried the obvious suspected fix, didn't work, root cause
   still open. This is why the default backend is `"egl"`, not `"osmesa"`.
3. Exact period-4 banding on non-square EGL renders, found while testing
   camera motion. Tested and REJECTED `sample_factor` (ray-march sample
   count) as the cause up to 10x -- pixel-identical regardless. Confirmed
   square is clean. **Worked around by rendering square and
   center-cropping** to the requested shape. You're independently digging
   into the actual root cause; this workaround is just so the add-on
   produces clean output in the meantime.
4. `known_uniforms.inc.glsl`'s own doc comment on the `viewport` uniform is
   wrong (says multiply by `.zw` for a screen UV; it's actually raw pixel
   width/height, you must divide) -- found while building the depth clip:
   this bug silently made ANY per-pixel-varying depth texture have zero
   effect (only a spatially-uniform test value "worked", by coincidence)
   until caught and fixed.
5. The EGL context's `config_attribs` never requested `EGL_ALPHA_SIZE` --
   real alpha bits -- so every render's alpha came back exactly `1.0`
   everywhere, background included, silently turning "transparent"
   background into opaque black through Alpha-Over compositing. Found while
   building the real end-to-end demo below (its first composited image had
   a black background instead of a transparent one); fixed by adding
   `EGL_ALPHA_SIZE, 8` to the config request, verified via the same demo
   (background alpha `1.0` -> `0.0`).

**The real end-to-end demo, per your request**: "a cycles render with a
sphere that is centered at a box corner with a radius of half the box... so
we can see the sphere and the volume... make sure the black background is
correctly addressed." `scripts/yt_idv_exploration/blender_scene/
render_real_demo.py` -- a genuine Cycles render (real lit/denoised sphere
geometry) composited with real yt_idv volume content via the add-on's
actual, unmodified `operators.render_current_frame()` (not a hand-built
`RenderRequest`, not a numpy composite) -- caught bug 5 above along the way.
Box edge length 2, sphere radius 1 (half the box) centered at corner
`(1, 1, 1)` -- close enough to the box center (`sqrt(3) ≈ 1.73` away) that
the sphere genuinely pokes into the box rather than missing it or
swallowing it whole. Result: `images/24_real_demo_sphere_and_volume.png` --
sphere and volume both visible, a genuine crescent-shaped gap at the
sphere's base where the real depth clip stops the ray march at the sphere's
own Cycles Z-buffer (not a flat cutout). Background transparency verified
two ways: a direct alpha check on a background pixel (`~0.0`), and
`images/25_real_demo_over_checkerboard.png` -- the same result composited
over a checkerboard, which shows through everywhere with no black rectangle
anywhere. The only non-real substitution: yt_idv still can't import into
Blender 5.3's own bundled Python 3.13 (confirmed by trying -- a compiled
dependency has no cp313 wheel), so the yt_idv call is bridged to the
throwaway venv via a subprocess for this one step; everything else in the
pipeline is the real, unmodified add-on code running against real Blender.

**You then asked about a light-pink patch showing up ON TOP of the sphere
itself** (distinct from the correct crescent-shaped gap around/below it) --
that led to two more real bugs, both in THIS ADD-ON's own code, both fixed:

6. **The one that actually mattered: `_render_with_yt_idv` called
   `scene.camera.update(**kwargs)` but never followed it with
   `scene.camera._update_matrices()`.** yt_idv's `TrackballCamera` overrides
   the hook `.update()`'s trait-change observer is supposed to call
   (`_compute_matrices`) with a no-op -- so `.update(fov=..., near_plane=...,
   ...)` sets those traits but never actually recomputes `projection_matrix`
   from them; yt_idv's own internal code always adds the explicit
   `._update_matrices()` call for exactly this reason, and ours was missing
   it. `view_matrix` happened to still look right (a lazy trait default,
   computed from whatever position/focus/up is current the first time
   something reads it), but `projection_matrix` silently kept a
   construction-time default (`fov=45`, `near_plane=0.001`) this ENTIRE
   project -- meaning every yt_idv render so far had the right camera
   position but the wrong field of view/near/far, without that being
   visually obvious until the depth clip exposed it as a full-opacity leak
   right on the sphere's own silhouette. Fixed by adding the missing call;
   confirmed by re-rendering the demo (clean sphere, no leak, and a visibly
   more accurate/tighter FOV than every earlier render in this exploration).
7. A smaller, independent issue found alongside it: `_registered_camera_
   kwargs`'s near/far-plane heuristic (`0.01x`/`4x` the distance to the
   domain center) produced a constant 400:1 near/far ratio regardless of the
   domain's real size, needlessly compressing its usable depth range near
   the far end (perspective depth is nonlinear). Fixed in `_near_far_planes`:
   near/far now bound the domain's own 8 unitary-cube corners directly.

See `yt_idv_renderer.py`'s module docstring for the full diagnosis story
(including how bug 6 was actually isolated: comparing `scene.camera.
projection_matrix` against a freshly-computed reference matrix before/after
the fix, not just eyeballing the final image).

**You then asked to demonstrate the OPPOSITE compositing direction: a
sphere partially occluded BY the volume**, rather than occluding it.
`scripts/yt_idv_exploration/blender_scene/render_partial_occlusion_demo.py`
-- needed no code change at all (the existing Alpha-Over compositing
already draws the volume over the Cycles beauty pass unconditionally using
the volume's own alpha), just the right scene geometry. First attempt
(sphere near the dataset's own density peak) hid the WHOLE sphere -- turned
out the current colormap's low-density opacity "floor" makes the entire box
interior read as nearly fully opaque past a shallow depth, so anything
placed inside the volume vanishes regardless of the peak. Fixed by
positioning the sphere BEHIND the box (past its far boundary along the
camera's view direction) and offset sideways, so part of its silhouette
overlaps the box (hidden under real volume color) while the rest sticks out
past its edge (fully visible) -- tuned with Blender's own
`world_to_camera_view` to check the sphere's projected screen position
against the box's. Verified against real pixel data, not just the picture:
a sample point expected to be hidden shows a large color difference from
its own Cycles-only render; a sample point expected to stay visible matches
its Cycles-only color EXACTLY (`|final - beauty| == 0.0000`). Result:
`images/32_partial_occlusion_sphere_and_volume.png` -- the sphere's real
material clearly visible on one side, clearly replaced by the volume's own
color on the other.

**You then asked for a new colormap** -- "fully transparent except at some
threshold value... ramp up the alpha to make it look foggy" -- for more of
this partial-occlusion demo work. Needed no new shader/uniform work:
`transfer_function.frag.glsl` already discards anything below `tf_min`
entirely, so the threshold IS just `tf_min`; the new part
(`_build_threshold_fog_transfer_function_data`) builds a continuous alpha
ramp instead of the existing mode's discrete isocontour layers, wired in as
a fourth render mode, `YT_IDV_THRESHOLD_FOG`. Two real things surfaced
while getting it working, both documented in full on `yt_idv_renderer.py`'s
own `TF_FOG_RAMP_POWER`/module docstring:
- **Another real yt_idv bug**: `passthrough.frag.glsl`, used as
  `block_rendering.transfer_function`'s own second pass, unconditionally
  forces `color.a = 1.0` -- discarding the first pass's real accumulated
  alpha, so the background composited as opaque black. Fixed with a new
  sibling shader (`passthrough_alpha.frag.glsl`) used ONLY for that one
  `second_fragment`, since the shared `passthrough.frag.glsl` is also used
  by several genuinely-always-opaque things (outlines, text, line plots)
  that shouldn't change. This also fixes the PRE-EXISTING isocontour
  transfer function mode's background-alpha correctness, not just the new
  one.
- **A real re-tuning need, not a bug**: your earlier camera-matrix fix
  (the missing `_update_matrices()` call, described above) changed which
  density values a given camera/scene actually samples; the OLD isocontour
  `TF_VALUE_MIN`/`MAX` now produce almost no signal at all. Re-tuned fresh
  constants for
  the new fog mode against a real scene (sweeping value_min/max/ramp_power,
  not guessing) -- found that a front-loaded ramp (`TF_FOG_RAMP_POWER <
  1.0`) recovers a strong, clearly visible result where a linear one
  couldn't. The OLD isocontour constants are flagged as needing the same
  re-tuning, not done here (see "What's next").

Result: `images/34_partial_occlusion_fog_sphere_and_volume.png` -- a soft,
glowing haze with a brighter core, genuinely translucent against a
checkerboard (`images/35_*`), clearly softer/hazier than MIP's harder color
swap in the demo above -- run via the SAME `render_partial_occlusion_demo.py`
script with `VOLUME_OVERRIDE_MODE=threshold_fog`.

**You then asked for a sphere at the domain center, zoomed in on a real
~10 kpc scale, with MIP itself showing transparency at the low end and
opacity at the high end.** Plain `max_intensity`'s own alpha turns out to
be a binary "was anything found" flag, totally decoupled from the value --
it can't do a value-driven ramp no matter the colormap. Fixed with a new
sibling shader (`apply_colormap_alpha.frag.glsl`, keeps the colormap's own
alpha instead of overriding it) and a new `YT_IDV_MAX_INTENSITY_ALPHA`
render mode, with a real alpha-ramped colormap
(`_build_mip_alpha_colormap_data`, reusing the fog mode's ramp shape)
assigned straight to `component.colormap.data`.

Getting the actual "zoom in" framing right took real empirical work, not a
guess: a camera merely a few multiples of 10 kpc away produced a bizarre
sharp TRIANGULAR artifact instead of gas structure -- turned out the camera
was pressed up against (or inside) a SINGLE AMR block's own triangulated
bounding cube at that distance, not resolving real density at all. Backing
off to 300-500 kpc (30-50x farther than the naive instinct) let the frame
span many nested blocks and the structure resolve cleanly -- confirmed by
comparing renders across 7 different distances. The 10 kpc marker sphere is
still there at 500 kpc, genuinely tiny (confirmed directly against the real
Cycles beauty pass) -- an honest answer to "zoom in on 10 kpc" at this
dataset's actual block resolution: the scale where structure resolves and
the scale of a 10 kpc marker don't cooperate at this dataset's max
refinement level. Swept 5 colormaps x 4 ramp powers against the same real
scene's cached depth data (no Cycles re-render per candidate) to settle on
final constants. Result: `images/36_zoomed_core_sphere_and_volume.png` -- a
soft glowing core (blue halo -> orange/white peak) with the tiny marker
sphere visible right at the center, genuinely transparent at the edges
(`images/37_*`). Full story, including the triangle-artifact diagnosis, in
`scripts/yt_idv_exploration/blender_scene/render_zoomed_core_demo.py`'s own
module docstring.

**You then said "I can't quite see it" about that marker sphere.** Three
real, distinct causes: (1) yellow was too close to "dusk"'s own warm
orange/white core color -- switched to saturated green; (2) `MIP_ALPHA_MAX`
was 1.0, letting the volume's own near-camera material build up to FULL
opacity in front of the sphere and hide it completely via Alpha-Over --
lowered to 0.7 (a real, general fix, not a one-off for this demo); (3) the
sphere itself was also enlarged, 10 -> 35 kpc. Re-rendered the same
`images/36_*`/`37_*` -- the green marker now reads clearly.

**You then asked to add the yt_idv/volume-renderer parameters to the
Blender panel itself, so you can experiment -- in a way that still lets the
add-on work with other renderers, and explicitly holding off on dataset
selection/loading for now.** Added `operators.YtIdvRenderSettings` (a new
`PointerProperty` on `VolumeOverridePreferences.yt_idv`) with a real,
editable field for every yt_idv module constant that shapes a render --
backend, both transfer functions' value ranges/colormaps/ramp shapes,
MIP-alpha's colormap/max/ramp power -- drawn in the Preferences panel ONLY
when `renderer_mode` is a `YT_IDV_*` option, so the panel stays exactly as
it was for `STUB`/`DEBUG_FLAT`/`SAMPLE_CONSTANT`. Deliberately did NOT add
dataset selection, per your instruction -- `SAMPLE_DATASET_NAME`/`SAMPLE_
FIELD` are still hardcoded.

Wiring reuses the EXACT mechanism `custom_params_extra` already established
for the generic per-object knobs, rather than growing `external_render.py`'s
frozen dataclass contract or making `yt_idv_renderer.py` depend on `bpy`:
`operators._yt_idv_custom_params()` merges the Preferences fields into
`VolumeObjectParams.custom_params` under `yt_idv_*`-prefixed keys (every
flagged object, every frame), and each yt_idv `_configure_*` function now
reads `custom_params.get("yt_idv_...", MODULE_CONSTANT)` -- falling back to
the exact same hardcoded default when the key's absent, so every
exploration script that builds its own `RenderRequest` by hand keeps
working completely unaffected. `yt_idv_renderer.py` still never imports
`bpy` or knows Preferences exist. Verified for real, not just via the unit
tests added alongside this (`_configure_transfer_function`/`_configure_
threshold_fog`/`_configure_max_intensity_alpha` monkeypatch-spy tests, plus
`test_operators.py`'s `addon_enabled` fixture using the REAL `bpy.ops.
preferences.addon_enable` flow, not just `bpy.utils.register_class` --
confirmed the latter alone does NOT populate `bpy.context.preferences.
addons[__package__]`, a real gotcha caught while writing this): changed
`prefs.yt_idv.fog_colormap`/`fog_ramp_power` in a real Blender session and
confirmed `build_volume_params(obj)`'s returned `custom_params` picked up
the new values immediately. Also directly invoked `VolumeOverridePreferences.draw()` against every
`YT_IDV_*` renderer_mode in real Blender to catch UI mistakes the pip-bpy
test suite can't (it never actually draws panels, so an invalid icon name
or bad `layout.prop` call would raise there, not in pytest) -- confirmed
clean on all four, including the `EXPERIMENTAL` icon identifier used for
the section header.

One acknowledged gap, not fixed: the exploration demo scripts' subprocess
bridge (`ytidv_worker.py`, used because yt_idv still can't import into
Blender 5.3's own Python) doesn't forward `custom_params` to the worker
subprocess -- those scripts hand-build their own `RenderRequest`s rather
than going through `operators.build_volume_params`, so Preferences-panel
overrides don't reach them. Real add-on usage (a real Blender with yt_idv
actually installed, calling `render_fn(request)` directly, no bridge) is
unaffected -- this is purely a limitation of this sandbox's own dev-only
bridge script, not the add-on.

Also demonstrated: a real Blender scene (flagged cube, orbiting camera,
reference wireframe boxes visualizing the dataset's AMR grid structure, a
composited marker sphere) driving the actual `operators.
build_camera_params()` across several frames -- see `scripts/
yt_idv_exploration/blender_scene/`. Caught two more small, real things along
the way: Empty objects never appear in a render (need real wireframe mesh
geometry instead), and `display_type='WIRE'` does NOT keep an object out of
Workbench's rendered output the way it looks like it should (`hide_render`
does, and matches the real add-on's own semantics for flagged objects
anyway) -- plus a large, dead-centered sphere reads as a flat sticker, not
"partly in front of" the content, until shrunk/offset.

Placeholder/debug renderer color (`VolumeOverrideProperties.color` and its
two fallbacks) is now a loud, obviously-synthetic red by default, not a
plausible pale blue -- your request, so placeholder output is unmistakable
against Cycles' own render while wiring things up.

**You then asked to wire the transfer function's alpha to a curve editor,
and potentially use a 2D image (functionally 1D) as the colormap** -- the
"idea floated for later" above, now actually built. Investigated first:
`PointerProperty(type=bpy.types.CurveMapping)` doesn't register at all on a
custom PropertyGroup (`TypeError: ... expected an RNA type derived from ID
or ID Property Group`, confirmed directly) -- worked around by piggybacking
on a hidden `ShaderNodeFloatCurve` node's own native `.mapping`, tucked into
an internal, never-shaded node group, and drawn via `UILayout.
template_curve_mapping(node, "mapping")` right inside the Preferences
panel -- confirmed this exists (only via `bl_rna.functions`, not plain
`dir()`) and takes exactly `(data, property)`. Added, for the two
continuous-ramp yt_idv modes (Threshold Fog / MIP Alpha -- not the
isocontour Transfer Function mode, whose shape doesn't reduce to one
alpha-vs-position curve the same way): an `alpha_source` toggle (Ramp
Power / Custom Curve) and a `color_source` toggle (Named Colormap / Image),
one shared setting driving both modes. The curve is sampled to a plain
list of floats (`operators._evaluate_alpha_curve`); the image is read back
via `image.pixels`, averaged across all rows ("functionally 1D" per your
own framing) and resampled horizontally (`operators.
_sample_image_as_1d_rgba`) to however many bins the LUT needs, regardless
of the image's own resolution/aspect. Both cross into `custom_params` under
new keys (`yt_idv_alpha_curve`/`yt_idv_color_image_rgba`); `yt_idv_
renderer.py`'s `_build_threshold_fog_transfer_function_data`/`_build_mip_
alpha_colormap_data` now take these as independent optional overrides
(`_resample_1d`/`_resample_rgb` handle the decoupling), falling back to the
pre-existing ramp-power/named-colormap mechanism exactly as before when
absent -- so this is a pure addition, not a behavior change, for anyone who
doesn't touch the new toggles.

Hit a SECOND, harder API wall partway through: an `Image` datablock
couldn't be a `PointerProperty` on `YtIdvRenderSettings` either --
confirmed `AddonPreferences` (and anything nested inside one, even a plain
PropertyGroup that merely contains an ID pointer somewhere in its tree)
categorically rejects ID/datablock `PointerProperty` fields (`ValueError:
... PointerProperty could not register because this type doesn't support
data-block properties`) -- a DIFFERENT restriction than CurveMapping's
(Image *is* a real ID type, unlike CurveMapping, but AddonPreferences still
can't hold one). Fixed with the standard workaround: store the image's
NAME (`color_image_name`, a plain `StringProperty`) and pick it via `layout.
prop_search(..., bpy.data, "images")` instead of `layout.prop`, looking it
up in `bpy.data.images` when building `custom_params` -- a stale name
(renamed/deleted image) falls back to the named-colormap mechanism silently
rather than raising mid-render.

Verified directly against a real Blender: curve creation/evaluation
(including that re-`initialize()`-ing an already-edited mapping doesn't
reset it -- confirmed by adding a point, evaluating, re-initializing, and
evaluating again with the same result), image pixel readback and
horizontal resampling on a hand-built test gradient, the full `custom_
params` round-trip for both the curve and image paths (present only when
selected, absent otherwise), stale-image-name handling, and `draw()`
succeeding with no exceptions across every `YT_IDV_*` renderer_mode with
both toggles in both positions. 21 new tests added (`test_yt_idv_renderer.
py`: resampling helpers, both `_build_*`/`_get_*` functions' curve/color
override behavior including cache-key correctness on content not object
identity, updated `_configure_*` spy tests; `test_operators.py`: node
creation/stability, curve evaluation, image sampling including the 0x0
guard, and the full custom_params flow for both mechanisms) -- 106/106
passing. **Not independently verified**: whether `template_curve_mapping`
actually renders/responds as an interactive widget in a real, on-screen
Preferences panel -- no interactive display was available in this
environment to check that, only that the API call itself is valid and
non-crashing; worth a quick look in your own session.

**You then asked: is it possible to animate the opacity with standard
keyframing?** Checked directly rather than guessing: **no**, not for any of
the yt_idv Preferences-panel alpha controls (Fog/MIP Alpha Max, Ramp Power,
the alpha curve added above) -- confirmed `AddonPreferences` instances have
no `animation_data` at all (`prefs.id_data` is `None`, `keyframe_insert`
raises `TypeError` outright) -- a hard, blanket Blender limitation on that
whole category of struct, not something fixable by moving fields around
within Preferences. Also checked whether the hidden curve's own control
points (which DO live on a real animatable ID, a `ShaderNodeTree`) could be
keyframed directly -- no: `keyframe_insert(data_path="location")` on a
curve point raises `could not make path to "location"`, a general Blender
limitation on curve widgets everywhere, confirmed not specific to this
add-on's setup.

Given that, added the thing that actually answers the question: a new
`obj.volume_override.opacity` field (default 1.0, on the Object itself --
a real animatable ID) applied by `operators._apply_opacity` uniformly to
every renderer's output, right after `render_fn(request)` returns,
regardless of `renderer_mode` -- multiplying the whole premultiplied-alpha
RGBA array by one scalar is mathematically exact for premultiplied alpha
(`rgb = color * alpha`, so scaling alpha by `f` scales premultiplied rgb by
the same `f`), so this is correct, not an approximation, and it's a
no-op copy-free fast path at the default 1.0. Verified end to end for
real: keyframed 0.0 at frame 1 -> 1.0 at frame 20 on a real Object,
`frame_set` to frame 10 read back 0.461 (correct linear interpolation) --
this is genuinely, actually keyframable today, unlike anything living in
Preferences. 3 new tests (`_apply_opacity`'s no-op and scaling behavior,
the field's keyframability) -- 109/109 passing.

## Known gaps / things to re-check

1. ~~**Blender 5.0+ compositor API mismatch.**~~ **Closed.**
   `compositor_build.py` now supports both the classic
   `scene.node_tree`/`CompositorNodeComposite` API (3.x/4.x) and Blender
   5.0+'s `scene.compositing_node_group`/`NodeGroupOutput` replacement,
   branching at runtime via `hasattr` checks, never a parsed version string.
   Real bugs found and fixed along the way (empirically confirmed, several
   against a real Blender 5.x install, not just the pip module):
   - Named-part EXRs breaking `bpy.data.images.load()` -- see "What
     happened when you actually ran it" #1 above; this is the same root
     cause as the row-order gap just closed above.
   - `img.colorspace_settings.name = "Linear"` (used when loading/writing
     every EXR this add-on touches) raises `TypeError` on Blender 5.0's
     default OCIO config -- that literal name doesn't exist there anymore.
     Fixed via `compositor_build.set_linear_colorspace()`, a small candidate
     list ending in the portable `scene_linear` OCIO role alias.
   - An Image node reading back a named-part EXR exposes its output socket
     under that part's name instead of the generic `"Image"` on Blender
     5.0+ -- silently returned `None` from `.outputs.get("Image")` and
     crashed the next link. Fixed via `_color_output_socket()`, falling
     back to `outputs[0]` (kept as defense in depth even though named parts
     are gone now).
   - No visible F12 result, and no way to see where a slow render was stuck
     -- see "What happened when you actually ran it" #2 and #3.
   Still not independently verified: the classic (<5.0) branch (unchanged
   by this port, but not re-confirmed either) -- see README "Blender 5.0+
   compositor" for the full writeup.
2. **Panorama fields location.** `operators._panorama_settings_source()`
   guesses `camera.data.cycles.panorama_type` (pre-5.0, probable) vs. core
   `Camera.panorama_type` (observed on the pip wheel, apparently a 5.0
   merge) -- untested against a real <5.0 Blender.
3. **VDB sequence frame resolution** (`operators.resolve_sequence_frame`)
   replicates Blender's documented start/duration/offset/clip-extend-
   repeat-pingpong semantics from the docs, not from a round-trip check
   against a real sequence.
4. **DOF focus-distance-from-object** resolves to Euclidean distance --
   not checked against what Cycles' own DOF math actually uses internally.

None of these block using the add-on; they're flagged so a wrong guess
doesn't get treated as settled fact.

## What's next

1. **yt_idv bug 3's root cause** (period-4 banding on non-square renders) --
   you're digging into this independently; `scripts/yt_idv_exploration/
   explore_bug3_banding.py` has the repro + the ruled-out sample_factor
   theory, ready to extend. Also visible as the residual jaggedness at the
   disc-clip boundary in the new Cycles-Z-buffer test images -- possibly
   related, possibly an AMR block-boundary artifact, not distinguished yet.
2. **Extend the real-depth-clip redo to the multi-frame orbit.**
   `render_real_demo.py` (above) redid the composite with the real depth
   clip for a single still frame; `render_frames_in_blender.py`'s
   camera-motion-over-frames + AMR grid-box wireframes still predates the
   Cycles Z-buffer wiring and uses the old flat alpha-over technique --
   worth redoing the same way across an animated camera path.
3. **Re-tune (not just "tune") the isocontour transfer function** --
   `TF_VALUE_MIN`/`MAX`/`TF_N_LAYERS`/`TF_COLORMAP` were found by trial and
   error for the OLD, pre-camera-fix registered camera; the threshold-fog
   investigation confirmed they now produce essentially no visible signal
   against the corrected camera (same root cause documented on
   `TF_FOG_RAMP_POWER` in `yt_idv_renderer.py` -- the fix changed which
   density values a given camera/scene combination actually samples).
   Consider deriving both this and the fog constants from the field's own
   statistics instead once a second dataset/field is in scope, rather than
   re-tuning by hand a third time.
4. **The Material-node-graph transfer function idea** (see above) --
   floated, not started.
5. **Wire in the real (non-yt_idv) renderer**: replace
   `external_render.render_volume()`'s body, if that's still a separate
   track from the yt_idv integration. Don't change the dataclass shapes
   without updating both sides of the boundary.
6. If a real Blender < 5.0 is ever in scope, re-run the classic-API branch
   of `compositor_build.py` at least once -- it's untouched by this
   session's port but wasn't re-exercised either.

See `README.md` for the full architecture writeup, all design-decision
rationale (why two render passes, why the Alpha-Over ordering, the
dataset/change-tracking design, the time-mapping/driver discussion, etc.),
and the four test-scene descriptions for verifying depth-clipped
integration once the real renderer is in.
