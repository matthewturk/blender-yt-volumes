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
| `src/cycles_volume_override/__init__.py` | `bl_info`, register/unregister |
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

**Idea floated for later, not started**: designing the transfer function
visually via a Blender Material node graph (e.g. a ColorRamp), baked to a
small image and fed in as `TransferFunctionTexture.data` -- plausible
mechanically, noted in `yt_idv_renderer.py`'s "Known rough edges".

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
3. **Tune the transfer function for real use** -- `TF_VALUE_MIN`/`MAX`/
   `TF_N_LAYERS`/`TF_COLORMAP` were found by trial and error for this one
   dataset/field; consider deriving them from the field's own statistics
   instead once a second dataset/field is in scope.
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
