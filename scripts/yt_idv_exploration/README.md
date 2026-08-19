# yt_idv exploration

Retained exploration for `src/cycles_volume_override/yt_idv_renderer.py`
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
   run is what caught yt_idv bug 5 (below). See "The real end-to-end demo"
   below -- this is the one worth reading if you only read one section.

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
