# cycles-yt

A Blender add-on that renders flagged objects with an external volume
renderer instead of Cycles: [yt_idv](https://github.com/yt-project/yt_idv),
the GPU volume renderer for [yt](https://yt-project.org) datasets. The volume
is depth-clipped against Cycles' Z pass and composited back in on F12 or
animation render, so Cycles geometry and simulation data share one camera
and occlude each other correctly. It handles deeply nested AMR data past
float32's precision, spherical data, time series, and yt transfer functions,
interactively in Blender or in batch from a config file. A companion Blender
extension, Deep Zoom (`extensions/deep_zoom`), authors camera paths through
deep hierarchies.

## Quick start

Needs Python 3.13 (the `bpy` module used for tests and batch renders is
Blender 5.2's), [uv](https://docs.astral.sh/uv/), and a GPU with EGL. yt_idv
comes from its `main` branch, which has the scene transforms, bindless
textures and transfer function conversion this relies on.

```
git clone https://github.com/matthewturk/cycles-override.git && cd cycles-override
PYTHONPATH= uv sync --group ytidv       # the add-on, bpy, yt, yt_idv
PYTHONPATH= uv run --group ytidv pytest -q
PYTHONPATH= uv run --group ytidv python examples/make_simple_blend.py examples/simple.blend
PYTHONPATH= uv run --group ytidv python -m cycles_yt.render_volume \
    examples/render_volume.toml --frames 1 --set blender.device='"CUDA"'
```

That renders frame 1 of `examples/render_volume.toml`, a 50 kpc region of
yt's HiresIsolatedGalaxy sample (downloaded on first use), into the simple
scene, to `renders/frame_0001.png`. Set `blender.device` to `OPTIX`, `CUDA`,
`HIP` or `CPU` for your GPU.

**In Blender:** zip `src/cycles_yt/` and install it
(Preferences > Add-ons > Install from Disk), and install yt and yt_idv into
Blender's own Python (`uv pip install --python <blender's python>
"yt_idv @ git+https://github.com/yt-project/yt_idv.git" pooch pandas h5py`). For deep camera paths, also zip `extensions/deep_zoom/`
and install it as an extension. Render Engine must be Cycles.

**Examples** (each script's docstring has the commands):

| | |
|---|---|
| `examples/render_volume.toml`, `render_series.toml` | a dataset or a time series into an existing .blend |
| `examples/popii_dive.toml` + `make_deep_zoom_shots_blend.py` | a saved yt_idv scene, a two-shot dive authored with Deep Zoom, yt shells |
| `examples/deep_zoom_shots.toml`, `deep_zoom_shots_tf.toml` | the same for the DeeplyNestedZoom sample, 12 decades |
| `examples/deep_zoom.toml` + `make_deep_zoom_blend.py` | a 12-decade dive with the volume's own zoom target/level |
| `examples/spherical_earth.toml`, `earth_spin.toml` + `make_earth_blend.py` | spherical data in an Empty-Sphere; time-varying |
| `examples/make_flythrough_blend.py` | a `flythrough_densest.py` path (not included) as a Deep Zoom scene |
| `examples/shells_tf.py` | yt transfer functions that follow the camera |

## Contents

```
src/cycles_yt/
    __init__.py            bl_info, register()/unregister()
    properties.py          per-object flag, Dataset Path, zoom target/level, UI panel
    operators.py           two-pass render orchestration, preferences, F12 takeover
    yt_idv_renderer.py     the yt_idv renderer: registration, zoom/shot space, depth clip, modes
    render_volume.py       batch rendering from a TOML config
    render_series.py       batch rendering of a time series
    external_render.py     the renderer interface contract (dataclasses)
    compositor_build.py    compositor graphs for the two passes
    scene_sync.py, change_tracking.py, handlers.py, sample_renderers.py
extensions/deep_zoom/      the Deep Zoom authoring extension (see its README)
examples/                  configs and scene builders (above)
scripts/                   verification tools and the original yt_idv exploration
tests/                     pytest suite
```

## Installing the add-on in Blender

This is a classic (`bl_info`-style) add-on, not a pip package Blender
installs for itself -- Blender doesn't consult `pyproject.toml`. Either:

- Zip `src/cycles_yt/` and use Blender's *Install from Disk*, or
- Symlink `src/cycles_yt/` into your Blender's addons directory
  and enable it in Preferences > Add-ons.

Then enable **cycles-yt** in Preferences.
Its preferences panel has only **Working Directory** (where per-frame EXR
intermediates go; default: system temp dir) and **Verbose Logging**.

The look lives on each flagged object's **Volume Override** panel (Object
Properties): the **Renderer** -- the yt_idv modes (Maximum Intensity,
Value-Driven Alpha, Projection, Isocontour and yt Transfer Functions,
Threshold Fog), or `SAMPLE_CONSTANT` (a non-physical debug renderer, see
"Sample renderer") -- and that mode's settings. They're saved in the .blend.

A flagged object (an Empty-Cube, or an Empty-Sphere for spherical data)
renders its **Dataset Path**: anything `yt.load` reads, or a scene saved with
yt_idv's `SceneGraph.save` (`.zip`); with no path, yt's IsolatedGalaxy
sample. The field defaults to density; set it with an extra custom param
`yt_idv_field` (e.g. `gas, temperature`).

## Sample renderer

`sample_renderers.constant_density_renderer()` is a worked example, not a
stand-in for the real renderer: it exercises the *entire* `RenderRequest`
contract -- generates real per-pixel rays from `CameraParams`, intersects
them against the volume's actual world-space bounding box (from
`VolumeObjectParams`), clips the resulting path length at the solid surface
(`DepthBuffer`, respecting its `depth_convention`/`row_order` rather than
assuming) -- but the "shading" is deliberately as dumb as the brief asked
for: **`output = clipped path length x a constant`**, no density
sampling, no marching, no scattering. It's there so there's something
visibly volume-shaped and *correctly occluded* to look at, and so the ray/
depth/position wiring is demonstrably exercised, before the real renderer
exists.

Pure numpy, deliberately importing nothing from `bpy` -- it only knows what
`RenderRequest` tells it, which is also why it's unit-testable
(`tests/test_sample_renderers.py`) without any Blender fixture for the
PANO/equirectangular path, and with only a thin bpy fixture for PERSP/ORTHO
(just to get a real, verified `projection_matrix` -- see "Camera projection").

Supports PERSP/ORTHO and PANO-equirectangular (the two camera paths with
empirically-confirmed formulas); any other panorama sub-type raises
`NotImplementedError` rather than silently rendering something wrong.
Select it via the volume's **Renderer** (`SAMPLE_CONSTANT`) to see it run
through the real two-pass pipeline, or call it directly on a hand-built
`RenderRequest` for a quick sanity check.

## yt_idv integration (exploratory)

`yt_idv_renderer.py` calls the REAL [yt_idv](
https://github.com/yt-project/yt_idv) volume renderer -- real offscreen
OpenGL, not a toy stand-in -- in five modes, all selectable via the
volume's **Renderer** setting: `YT_IDV_MAX_INTENSITY` (Maximum Intensity
Projection), `YT_IDV_TRANSFER_FUNCTION` (a real transfer function --
isocontour layers at fixed densities in log space,
`yt.ColorTransferFunction.add_layers`, not a placeholder colormap),
`YT_IDV_PROJECTION` (unweighted line-of-sight integration -- tends to look
smoother/less speckled than MIP for the same data), `YT_IDV_THRESHOLD_FOG`
(a continuous, hazy alpha ramp above a density threshold), and
`YT_IDV_MAX_INTENSITY_ALPHA` (MIP with a value-driven, not presence-driven,
alpha). By default all five render one hardcoded sample dataset
(`yt.load_sample("IsolatedGalaxy")`) -- but `custom_params["yt_idv_
volume_provider"]` (set via `obj.volume_override.yt_idv_volume_provider`,
a `"module.submodule:function_name"` string) replaces that with a Python
function YOU write, called once per object per frame, stateful across
calls -- see `yt_idv_renderer.py`'s "Custom volume setup" for the full
harness. See that module's own docstring generally for the full, current
account of what is and isn't real -- this section just orients you and
covers installation.

**All three now depth-clip against Cycles' real Z pass, and the camera's
real position (not just orientation) carries over too** -- both use the
flagged object's own `bounding_box_local`/`matrix_world` (already part of
`VolumeObjectParams`, no new field needed) as the registration between
Blender world space and yt_idv's unitary domain: a unitary point lives at
`bbox_min + u*bbox_extent`, then `matrix_world` into Blender world space.
yt_idv itself CAN stop a ray's integration early at an externally-supplied
per-pixel depth (`yt_idv_renderer.set_external_depth_clip` -- required a
real, small yt_idv shader modification, see "Four real yt_idv bugs" below)
-- Cycles' own Z-buffer gets reconstructed into world-space hit points
(reusing the exact ray math already verified in
`sample_renderers._persp_ortho_rays`), transformed through the registration
into unitary space, then projected through yt_idv's own
now-correctly-positioned camera to get its window-space depth convention.
No new yt_idv Camera subclass needed -- `BaseCamera.update(position=...,
focus=..., up=..., ...)` was already general enough, it just needed
genuinely-transformed values fed in instead of an orbit-at-a-fixed-distance
heuristic. Still approximate in two specific, undisguised ways: yt_idv
renders square while Blender's camera essentially never is (so depth
reconstruction happens on Blender's real pixel grid, then gets
center-padded/cropped to align with the square canvas -- a square-aspect
camera can't reproduce Blender's true horizontal FOV), and the object's
bounding box has to actually correspond to where you want the dataset
positioned (a registration you place by sizing/moving the flagged object,
not inferred from the dataset). Verified against a real, disc-shaped
fabricated Cycles Z-buffer producing a genuine disc-shaped hole in the
render, AND the underlying math separately verified against a real
bpy-computed `projection_matrix` in `tests/test_yt_idv_renderer.py` -- see
`scripts/yt_idv_exploration/explore_cycles_zbuffer_depth_clip.py`.

### Installing yt_idv into your Blender's Python

Real Blender's own embedded Python is NOT the same Python as this repo's
`bpy==5.2.0` pip wheel dev venv (see "Development & testing" below) -- it
doesn't have that wheel's version constraints, so installing `yt_idv` there
is a normal "add a third-party package to Blender" operation: find
Blender's bundled Python (e.g. `blender --python-expr "import
sys;print(sys.executable)"`, or the `python.exe`/`python3.1x` under
Blender's own install directory) and `pip install` into it directly (using
that interpreter's own `-m pip`, ensuring pip first via `-m ensurepip` if
needed) -- see Blender's manual, "Installing Python Modules", for the
general mechanism this relies on. Install `yt_idv` itself plus `pooch`/
`pandas`/`h5py` (lazy dependencies `yt.load_sample`'s Enzo/IsolatedGalaxy
path needs that aren't part of `yt_idv`'s own declared dependencies).

**On Python 3.13+ (e.g. Blender 5.x's own bundled Python), install yt_idv
from git for now, not `pip install yt_idv` / PyPI.** PyPI's latest release
(0.5.4, as of this writing) lists `imgui` as an unconditional dependency,
which has no cp313 wheel and fails to build from source against 3.13's
changed C API -- this is exactly what used to make `yt_idv` uninstallable
into a modern Blender's own Python at all (see "Real yt_idv bugs" below,
and the now-corrected claim this repo's own `yt_idv_renderer.py` module
docstring used to make about that). Fixed upstream 2026-09-15
(yt-project/yt_idv#268, "imgui-optional"): imgui moved to an opt-in `gui`
extra, needed only for the interactive pyglet+imgui control panel -- never
for this add-on's own `egl`-backend, no-GUI rendering path -- so installing
without it no longer touches imgui at all. Not yet in a PyPI release
(0.5.4 predates the fix), so for now:

```
pip install "git+https://github.com/yt-project/yt_idv.git" pooch pandas h5py
```

into Blender's own Python, or use its `test`/`gui` extras if you also want
yt_idv's own test suite or the interactive GUI (`pip install
"yt_idv[gui] @ git+https://github.com/yt-project/yt_idv.git"` -- imgui does
build fine on Python <=3.12, just not 3.13+). Once a release ships the fix,
a plain `pip install yt_idv` will work the same way. This repo's own dev
venv installs yt_idv the same way, via the optional `ytidv` dependency
group -- see "Development & testing" below; it's no longer a separate
Python/venv story from your real Blender's install, just a separate
`pip`/`uv` invocation.

**Use yt_idv `main`.** The external depth clip (yt-project/yt_idv#247), scene
transforms for subset scaling and deep zooms (#287), bindless textures (#285)
and the transfer function conversion (#288) are all newer than the latest
release. Older versions import and render, but `set_external_depth_clip` and
`scale_scene_to_subset` raise rather than silently doing nothing.

### Rendering from a config file

`render_volume` renders a yt dataset into an existing `.blend` without a
custom harness: one TOML file names the dataset (plus optional subset), the
`.blend` file, the Empty-Cube object to replace, and the look.

```
PYTHONPATH= uv run --group ytidv python -m cycles_yt.render_volume examples/render_volume.toml
PYTHONPATH= uv run --group ytidv python -m cycles_yt.render_volume \
    --sample IsolatedGalaxy --blend scene.blend --volume "Volume 1" --frames 1-10 \
    --set data.subset.kind='"sphere"' --set data.subset.radius='[20, "kpc"]'
```

`examples/render_volume.toml` documents every key. Options override the
file, `--set section.key=value` reaches any key (values parse as TOML), and
`--check` prints the merged config without rendering. `[render]` keys
override the volume's own settings in the `.blend`; with no `[render]` the
`.blend` renders exactly as authored in the GUI. Each run gets its own
temporary `work_dir` unless one is set, so concurrent runs don't collide.

### Deep zooms past float32

Moving a Blender camera toward a tiny feature normally stops working around
1e-6 of the box: Blender holds positions in float32, and next to an ordinary
world offset there is no room left for the camera's offset from the target.
The volume object's **Zoom Target** (`"max"` or `"x, y, z"` in unitary
coordinates, a string so it keeps float64 precision) anchors two ways around
it; `[zoom]` in a config sets it.

- **Magnify** (`examples/deep_zoom.toml`, built by
  `examples/make_deep_zoom_blend.py`): the camera moves at ordinary scales
  and **Zoom Level** (a power of 10, animatable; `level` in `[zoom]` is a
  `[[frame, level], ...]` curve) magnifies the data about the target. Cycles
  objects don't magnify with it.
- **Fly**: put the target at the world origin, where float32 keeps its
  relative precision at any scale, and fly the camera in for real. With the
  target at the origin Blender holds the camera to ~1e-7 of its distance at
  1e-10 box widths; parented 1 unit away, it is 1.5% off at 1e-6 and lost by
  1e-8. Cycles objects near the target stay registered with the data, though
  Blender clips them closer than 1e-6 world units to the camera.

Either way the worker rescales yt_idv each frame so the camera is about one
unit from the target in model space, and maps world to model coordinates
relative to the target in float64. `examples/make_flythrough_blend.py` turns
`flythrough_densest.py`'s camera paths into either kind of scene (`--fly`).

### Authoring deep camera paths: the Deep Zoom extension

`extensions/deep_zoom` is a self-contained Blender extension for designing
camera paths through deeply nested data. It works in shot space, a float64
anchor at the world origin plus an animatable zoom, so the camera stays at
ordinary sizes at any depth. It provides shots, a grid map of the AMR
hierarchy, re-anchor and rescale cuts, a path check that says where they're
needed, and float64 path export. See its README. This renderer picks up its
scene settings automatically; `examples/deep_zoom_shots.toml` renders a
two-shot dive into DeeplyNestedZoom authored by
`examples/make_deep_zoom_shots_blend.py`.

### yt transfer functions

`render.mode = "YT_IDV_YT_TRANSFER_FUNCTION"` renders with a transfer function
you build in yt, a `ColorTransferFunction` with grey opacity over log10 of
the field. It is converted with yt_idv's `tf_conversion` (yt-project/yt_idv#288),
so it renders as yt would. Name the function in `[render.yt_idv]`:

    yt_tf = "my_tf.py:make"      # or "module:function"
    yt_tf_ray_length = 0.03      # divides emission and opacity
    yt_tf_sample_factor = 8      # thin layers need finer ray steps

`make()` with no arguments is converted once. `make(context)` is called every
frame with `frame`, `camera` (the camera's unitary float64 position) and `ds`,
so a transfer function can follow the camera. `examples/shells_tf.py`
hides isodensity layers sparser than the gas around the camera, which turns
the DeeplyNestedZoom dive (`examples/deep_zoom_shots_tf.toml`) into a passage
through nested shells, one per decade of zoom. In a zoom the near and far
planes are bounded to 1e-3 and 1e3 times the camera's distance to the target:
yt_idv rebuilds rays from the inverse projection in float32, and a domain-wide
far plane there bands transfer-function renders.

### Spherical data

For a dataset in yt's spherical geometry, flag an **Empty with Display Type
'Sphere'** instead of a cube. yt_idv draws spherical data in cartesian
coordinates scaled by the domain's outer radius, so the Empty's radius is the
data's outer edge (not necessarily the planet's surface), its local +z is the
pole (theta = 0) and its local +x is longitude 0 (phi = 0). Keep its scale
uniform. Zoom and subset scaling don't apply: yt_idv can't rescale spherical
data.

`examples/spherical_earth.toml` renders a synthetic mantle with plumes under
Hawaii, Iceland and Reunion (`examples/spherical_earth.py`), inside a globe
built by `examples/make_earth_blend.py`: an opaque core, a graticule and rings
at the hotspots, all Cycles geometry the volume is depth-clipped against. It
loads the dataset with `data.loader`, which names any `"module:function"` or
`"file.py:function"` that returns a yt dataset.

### Animating a dataset series

`render_series` drives `render_volume` over a time series. The config is the
same, except `[data]` names a `series` (glob or list of paths) or a
`simulation` (`yt.load_simulation` parameter file and code), and `[time]`
maps frames to outputs: by simulation time or by output index, with optional
`[frame, time]` keyframes to stretch or hold. `examples/render_series.toml`
has the details.

```
PYTHONPATH= uv run --group ytidv python -m cycles_yt.render_series examples/render_series.toml --dry-run
```

Consecutive frames that show the same output form a segment, and each segment
runs as its own `render_volume` process, because yt_idv never frees GPU
textures within a process. Each segment's resolved config is saved under
`<output dir>/.render_series/`, so one segment can be rerun by hand;
`--skip-existing` resumes an interrupted run. `cmap_range = "first_last"`
auto-scales one probe frame each of the first and last output and
interpolates the colormap range between them over the frames, so colors
change smoothly as structure forms instead of pumping between outputs or
saturating against the first output's range (`"first"` pins that, and a
`[[frame, min, max], ...]` curve sets it by hand). yt_idv colormaps are
log-scaled by default, so set `cmap_interpolation = "log"`; linear
interpolation keeps the min near the first output's for most of the run. A
resumed run reuses the saved range. A `[data.subset]` is re-evaluated per output (a `"max"`
center can jump between outputs), and subset scaling follows each output's
blocks, so the stretch can change between segments.

### Rendering a subset of a dataset

A volume provider can pass any yt data object to `add_scene` -- `ds.region`,
`ds.sphere`, `ds.cut_region` -- and the Blender volume box still maps to
yt_idv's unitary `[0, 1]^3`. Unscaled, the subset sits at its true position and
size inside that cube, so it looks small. `yt_idv_renderer.
scale_scene_to_subset(scene)` (needs #287) stretches the scene so the
subset's longest axis fills the cube, starting at its origin corner; shorter
axes leave empty space at their high end. It leaves full-domain scenes alone.
`scene_setup.py`'s `SUBSET_HALF_WIDTH_KPC`/`SCALE_SUBSET` show both.

The stretch uses the data object's own bounding box; grid blocks that stick
out past it are masked off by yt_idv's selector bitmap. Projection values
also grow by the stretch factor, so fixed colormap ranges
from an unscaled run don't carry over.

### Six real yt_idv bugs found while wiring this up

All found by testing NON-square resolutions -- every one of yt_idv's own
examples/tests uses a square one (`width=height=1024`), so these went
unnoticed upstream. Full detail, reproduction scripts, and reference images
in `scripts/yt_idv_exploration/` (retained, not just described):

1. **Wrong axis order.** `yt_idv.scene_graph.SceneGraph.image` returns
   `(width, height, 4)` for non-square renders instead of the conventional
   `(height, width, 4)`, because its own corrective `arr.swapaxes(0, 1)`
   call discards the return value -- a complete no-op. Worked around
   entirely on this add-on's side (`_fix_yt_idv_axis_order`); no yt_idv
   edit needed.
2. **OSMesa backend corrupts non-square renders** (not just mis-shapes them
   -- genuinely tiled/repeated garbage). The obvious suspected fix
   (`OSMesaMakeCurrent`'s width/height argument order) did NOT resolve it;
   real cause still open. This is why `yt_idv_renderer.DEFAULT_BACKEND` is
   `"egl"`, not `"osmesa"`, despite EGL needing a working GPU + driver
   stack that OSMesa doesn't.
3. **Exact period-4 banding on non-square EGL renders**, found while
   testing camera motion. Confirmed via a raw scanline dump (`[value, 0, 0,
   value]` repeating); tested and REJECTED `sample_factor` (ray-march
   sample count) as the cause up to 10x -- pixel-identical regardless.
   Confirmed a square render of the same scene/camera is clean. Root cause
   still open (candidates: EGL surface/multisample config, a width/height
   mixup in yt_idv's own viewport or grid-expansion geometry shader for
   non-square viewports specifically). **Worked around by rendering square
   and center-cropping** to the requested shape (`yt_idv_max_intensity_
   renderer`'s own docstring has the mechanics and cost).
4. **A wrong doc comment on yt_idv's own `viewport` uniform.**
   `known_uniforms.inc.glsl` documents `viewport` as `(offset_x, offset_y,
   1 / screen_x, 1 / screen_y)` (multiply by `.zw` for a UV); it's actually
   set from the raw `GL.glGetIntegerv(GL_VIEWPORT)` -- `(x, y, width,
   height)` in pixels (divide, don't multiply). Found while adding the
   external depth clip below: multiplying (as the comment says) sends the
   computed screen UV far outside `[0, 1]`, and with the texture's default
   "clamp" boundary, every fragment samples the same corner texel
   regardless of its real position -- invisible for a spatially-uniform
   test value, but a per-pixel-varying one had zero effect anywhere until
   this was found.
5. **The EGL context requests no alpha bits at all.** `egl_context.py`'s
   `config_attribs` asks for 8 bits each of red/green/blue but never
   `EGL_ALPHA_SIZE` -- so the framebuffer has no real alpha channel, and
   every pixel read back via `glReadPixels(..., GL_RGBA, ...)` comes back
   with alpha exactly `1.0`, even genuinely-empty background pixels (RGB
   correctly `(0, 0, 0)` there). This silently turned "transparent"
   background into opaque black through Alpha-Over compositing. Found while
   verifying the real end-to-end sphere-and-volume demo's background
   (below) -- fixed by adding `EGL_ALPHA_SIZE, 8` to the config request;
   confirmed via the same demo, background alpha going from `1.0` to `0.0`.
   (Since independently fixed upstream too, on `main` -- the local `yt_idv/`
   checkout's own rebase onto current `main` picks up the identical fix
   natively now, so this repo no longer carries it as part of its own
   patch.)
6. **`passthrough.frag.glsl` (used as `block_rendering.transfer_function`'s
   own second pass) unconditionally sets `color.a = 1.0`**, discarding the
   first pass's real per-pixel accumulated alpha entirely -- so any
   `"transfer_function"`-render-method render (isocontours, or the new
   threshold-fog mode below) composited its background as opaque black
   instead of transparent. Not fixed by editing that shared shader in place
   (several OTHER usages -- outlines, text, line plots -- genuinely want
   always-opaque); fixed by adding a new sibling shader,
   `passthrough_alpha.frag.glsl` (same, minus that one line), and pointing
   only `block_rendering.transfer_function`'s `second_fragment` at it.

None of these are this add-on's bugs to fix, but they're load-bearing for
`yt_idv_renderer.py`'s design decisions (backend choice, the axis-order
correction, the square-render-then-crop approach, the depth-clip shader
addition) -- worth real yt_idv fixes and upstream bug reports before this
graduates past exploration.

### External depth clip -- a real, small yt_idv shader modification

GLSL can't be monkeypatched the way Python can, so making a solid occluder
actually stop a ray's integration early (rather than an all-or-nothing flat
cutout) required editing `yt_idv/shaders/ray_tracing.frag.glsl` +
`known_uniforms.inc.glsl` + `yt_idv/scene_components/blocks.py` -- see
`scripts/yt_idv_exploration/yt_idv_external_depth_clip.patch` for the exact
diff. New uniform pair, off by default, zero effect on existing behavior
when unused. Verified with a real per-pixel, sphere-shaped depth (analytic
ray/sphere intersection): a genuine, clean partial "bite" taken out of the
rendered nebula, gas outside the sphere's silhouette rendering completely
normally -- something the flat alpha-over composite below can never
produce. Full story, including bug 4 above, in `yt_idv_renderer.py`'s
"External depth clip" docstring section and `scripts/yt_idv_exploration/
explore_external_depth_clip.py`.

**Status upstream:** this is yt-project/yt_idv#247 ("Fixing for external
usage"), now merged into `main`. Without it, `set_external_depth_clip` raises
rather than silently doing nothing (no component would have
`use_external_depth_clip`, and every render would still look plausible).

### Threshold fog transfer function -- a fourth render mode, `YT_IDV_THRESHOLD_FOG`

Requested directly: "a colormap that is fully transparent except at some
threshold value... ramp up the alpha to make it look foggy." Needed no new
shader/uniform work -- `transfer_function.frag.glsl` already discards any
sample below `tf_min` entirely, so setting `tf_min` to the threshold gives
"fully transparent below this" for free; the new part is
`_build_threshold_fog_transfer_function_data` building a CONTINUOUS ramp
(alpha rising from 0 at the threshold up to `TF_FOG_ALPHA_MAX`, front-loaded
via `TF_FOG_RAMP_POWER < 1` so it looks genuinely hazy rather than a hard
line) instead of the existing mode's discrete isocontour layers. Finding
good `TF_FOG_VALUE_MIN`/`MAX`/`RAMP_POWER` values took real, scene-specific
tuning (see `yt_idv_renderer.py`'s own comment on `TF_FOG_RAMP_POWER` for
why) -- and building the real end-to-end demo with it caught bug 6 above
(background compositing as opaque black) along the way. Result:
`images/34_partial_occlusion_fog_sphere_and_volume.png` -- a soft, glowing
haze with a brighter core, genuinely translucent (confirmed against a
checkerboard, `images/35_*`) rather than MIP/projection's harder,
near-uniform floor fill.

### Maximum Intensity with a value-driven alpha -- a fifth render mode, `YT_IDV_MAX_INTENSITY_ALPHA`

Requested directly: zoom in on a real physical scale (10 kpc) and see MIP
with transparency at the low end, opacity at the high end. Plain `max_
intensity`'s own alpha is a binary "was anything found" flag, completely
decoupled from the selected value -- it can't do this regardless of the
colormap. Fixed with a new sibling shader, `apply_colormap_alpha.frag.glsl`
(keeps the colormap's own alpha channel instead of overriding it) and a new
`max_intensity_alpha` combo, plus a real alpha-ramped colormap
(`_build_mip_alpha_colormap_data`, reusing the fog mode's own ramp shape)
assigned directly to `component.colormap.data`.

Getting the "zoom in on 10 kpc" framing right took real, empirical
investigation, documented in full in `scripts/yt_idv_exploration/
blender_scene/render_zoomed_core_demo.py`: a camera merely a few multiples
of 10 kpc away produces a sharp, geometric TRIANGULAR artifact instead of
organic gas structure (at that distance the camera sits against/inside a
SINGLE AMR block's own triangulated bounding cube -- not resolved density
at all), while backing off to 300-500 kpc lets the frame span many nested
blocks and the structure resolves cleanly. The 10 kpc marker sphere is
still there at 500 kpc, genuinely tiny -- an honest answer to "zoom in on a
10 kpc feature" at this dataset's actual block resolution, not a failure to
zoom. Result: `images/36_zoomed_core_sphere_and_volume.png` -- a soft
glowing core (dusk colormap) with the marker sphere visible right at the
peak, genuinely transparent at the edges (`images/37_*`).

**Follow-up: "I can't quite see it."** The first version's marker sphere
(10 kpc, bright yellow) was genuinely hard to spot. Three real, distinct
causes, all fixed:
1. Yellow sat too close to the "dusk" colormap's own warm orange/white
   core color -- the sphere blended into the gas around it instead of
   contrasting. Switched to a saturated green, a hue "dusk" never produces.
2. `MIP_ALPHA_MAX` was 1.0 -- the volume's own near-camera material could
   build up to FULL opacity before the ray ever reached the sphere, hiding
   it completely via Alpha-Over regardless of its own real depth-clip
   visibility. Lowered to 0.7 so something behind the densest visible gas
   always shows through -- this is a real, general fix (affects every
   `YT_IDV_MAX_INTENSITY_ALPHA` render, not just this one demo), not a
   one-off tweak.
3. The sphere itself was also enlarged, 10 -> 35 kpc, as a third,
   complementary fix -- still tiny relative to the ~500 kpc frame.

Re-rendered result (same `images/36_*`/`37_*` paths, overwritten): the green
marker now reads clearly against the orange/white core in both the plain
and checkerboard exports.

### The real end-to-end demo: a real Cycles sphere composited with real yt_idv volume content

`scripts/yt_idv_exploration/blender_scene/render_real_demo.py` is the whole
pipeline, run for real: a genuine Cycles render (real BSDF material, real
sun lamp, 64 samples, denoised) of a sphere embedded inside the flagged
volume box, composited with real yt_idv volume content -- via the add-on's
actual, unmodified `operators.render_current_frame()` two-pass pipeline,
not a hand-built `RenderRequest` and not a numpy composite. Runs entirely
in-process now (real `bpy` + real, patched `yt_idv` in one Python 3.13
interpreter, via this repo's own `uv`-managed dev venv -- see "yt_idv
integration" and "Development & testing" above) -- no subprocess bridge, no
monkeypatched `RENDERER_MODES`, `operators.render_current_frame()`
unmodified end to end.

**Where the sphere sits matters, and it took two more tries to get right.**
Originally centered at a box CORNER, radius = half the box's edge:

1. **At the corner FARTHEST from the camera** -- the very first version --
   left little to no volume density actually behind the sphere for the
   depth clip to remove: the box's own domain had already mostly ended
   right around there. Confirmed directly by `verify_overlap.py` (below):
   disabling the depth clip changed NOTHING, even in `projection` mode.
2. **Moved to the NEAREST corner instead** -- fixed the clip (a real,
   measurable 100% alpha drop where the sphere was), but combined with a
   sphere half as wide as the box, it ended up occluding the box's ENTIRE
   visible silhouette from the camera: no volume showed anywhere, and
   there was no near-side volume left in front of the sphere either (it
   sat right at the box's own near face), so the sphere itself always
   looked fully opaque -- occlusion only ever ran ONE direction.

**Fixed by shrinking the sphere and embedding it more centrally**, off
toward the camera but not poking past the box's near face: now genuine
volume shows on BOTH sides of it along camera rays -- some in front
(ordinary Alpha Over veils part of the sphere in a soft haze, no depth clip
involved) and some behind (the real depth clip removes it, so the sphere
still visibly occludes something too) -- while the box's own silhouette
stays visible all around it. Both occlusion directions visible in the same
image is the actual point: proof the two objects genuinely share the same
3D space, not just that one always wins.

Result: `images/24_real_demo_sphere_and_volume.png` -- the sphere renders
as real lit Cycles geometry, clearly a solid ball (not washed out), with
real gas from the volume veiling part of its surface and clean depth-
clipped gas visible around/behind it -- not a flat, all-or-nothing cutout,
and not two disconnected shapes floating near each other. This run is also
what caught yt_idv bug 5 above: the background initially composited as
opaque black instead of transparent, traced to the EGL context's missing
alpha bits. Fixed, then verified two ways: a direct alpha-value check on a
background pixel (`~0.0`), and `images/25_real_demo_over_checkerboard.png`,
the same result composited over a checkerboard -- the checker pattern shows
through everywhere neither the sphere nor the volume is present, which an
opaque background could not produce.

### Verifying the sphere and volume actually interact, not just look plausible

A composited image that merely LOOKS plausible isn't proof the depth clip
did anything -- see "External depth clip" above for a real case where it
silently did nothing while still producing a normal-looking render.
`scripts/yt_idv_exploration/blender_scene/verify_overlap.py` is the actual
test: it renders the SAME real scene through the SAME real pipeline TWICE
per renderer mode -- once with the depth clip enabled (normal), once
forced off (`yt_idv_renderer.set_external_depth_clip` monkeypatched to
always pass `depth=None`, its own documented "disable" contract) -- and
diffs the two. The sphere's own Cycles beauty pass never touches the depth
clip at all (rendered once in pass 1, before the volume renderer even
runs), so it's checked bit-identical between the two runs first, as a
sanity check on the comparison itself; only the volume layer's own alpha,
specifically inside vs. outside the sphere's on-screen silhouette (read
straight from that same beauty pass's alpha channel -- the sphere is the
only opaque geometry in the scene), is the actual signal.

With the final, embedded sphere placement above, on both `projection` and
`max_intensity`:

```
mean volume alpha INSIDE sphere silhouette:  unclipped=1.0000 clipped=0.4010  drop=0.5990 (59.9%)
mean volume alpha OUTSIDE sphere silhouette: unclipped=0.2119 clipped=0.2119  drop=0.0000
```

-- a ~60% drop exactly where the sphere is, 0% everywhere else:
spatially-localized, not a global effect, and PARTIAL this time -- some of
the sphere's silhouette already had near-side volume saturating to full
alpha before the clip point (no room left for the clip to remove anything
more there), the rest had thin/low near-side alpha, so clipping away the
far-side volume there makes a big difference. See
`images/29_verify_overlap_diff_projection.png`, a per-pixel heatmap where
red is "alpha the clip removed" and a faint green outline traces the
sphere's silhouette for registration: the red lobes line up almost exactly
with the reddish-brown UNVEILED parts of the sphere in image 24, and the
dark (no-drop) regions line up with the pink-VEILED parts -- the same
story, told two ways. **This still does NOT distinguish MIP from
`projection`** -- both happen to show the same 59.9% drop with this exact
geometry. MIP keeps only the single brightest sample along each ray and
discards where along the ray it came from, so a PARTIAL depth occlusion
(the far-side clip removing only SOME of a ray's density, not all of it)
is exactly the case where MIP's own blind spot COULD show up (the peak
sample surviving the clip either way) -- it just doesn't happen to here.
Demonstrating that asymmetry for real would need a geometry specifically
tuned so the density peak sits on the near side of the clip plane for some
pixels; not attempted here.

**Two more real bugs found via this same demo, both in THIS ADD-ON's own
code (not yt_idv), both fixed:**

1. **The actual one that mattered: `_render_with_yt_idv` never called
   `scene.camera._update_matrices()` after `scene.camera.update(...)`.**
   `yt_idv.cameras.trackball_camera.TrackballCamera` overrides
   `_compute_matrices` (the hook `.update()`'s own trait-change observer is
   supposed to call) to a literal no-op, so `.update(position=..., fov=...,
   near_plane=..., ...)` alone sets those traits but never recomputes
   `projection_matrix`/`view_matrix` from them -- yt_idv's own internal code
   always follows `.update()` with an explicit `._update_matrices()` for
   exactly this reason. Effect: `view_matrix` "accidentally" still looked
   right (its own lazy default computes from whatever position/focus/up is
   CURRENT the first time something reads it, and nothing read it before our
   `.update()` call), but `projection_matrix` silently kept a
   construction-time default (`fov=45`, `near_plane=0.001`) -- meaning every
   yt_idv render this project has ever produced had the right camera
   POSITION but the WRONG field of view and near/far planes, invisibly,
   the entire time it worked on this project. First appeared as a
   full-opacity "leak" through the depth clip right on top of the real
   Cycles sphere's own silhouette; fixed by adding the missing call.
   Confirmed: re-rendering the demo after the fix produces a clean sphere
   with no leak, and a visibly tighter/more accurate field of view than
   every earlier render in this whole exploration.
2. A smaller, independent precision issue in the SAME area:
   `_registered_camera_kwargs`'s near/far-plane heuristic (`0.01x`/`4x` the
   distance to the domain CENTER) produced a constant 400:1 near/far ratio
   regardless of the domain's actual size, needlessly compressing the
   domain's usable depth range near NDC depth 1.0 (perspective projection is
   nonlinear in depth). Fixed in `_near_far_planes`: near/far are now sized
   around the domain's own 8 unitary-cube corners instead.

See `yt_idv_renderer.py`'s module docstring for the full account of both,
including exactly how the first one was diagnosed (comparing
`scene.camera.projection_matrix` against a freshly-computed
`get_perspective_matrix(...)` before/after the fix).

### The reverse demo: a real sphere partially occluded BY the volume

`scripts/yt_idv_exploration/blender_scene/render_partial_occlusion_demo.py`
demonstrates the opposite compositing direction from the demo above: there,
a solid sphere occluded the volume; here, the volume occludes part of a
solid sphere. No code change was needed for this -- the add-on's existing
Alpha-Over compositing already draws the volume layer over the Cycles
beauty pass unconditionally, using the volume's own alpha -- only a scene
geometry that puts a sphere partly behind real, opaque-looking volume
content, from the camera's point of view.

The first geometry attempt (sphere centered near the dataset's own density
peak) hid the ENTIRE sphere, not just the part near the peak -- it turned
out the current colormap's low-density opacity "floor" (already documented
below) makes the WHOLE box interior read as nearly fully opaque past a
modest depth into the domain, so anything placed meaningfully inside the
volume disappears regardless of the peak. The fix: position the sphere
BEHIND the box (past its far boundary along the camera's view direction)
and OFFSET SIDEWAYS, so part of its silhouette overlaps the box's own
screen footprint (hidden under real volume color) while the rest sticks
out past the box's edge (fully visible, zero volume alpha there) -- tuned
using Blender's own `world_to_camera_view` to check the sphere's projected
screen columns against the box's. Verified directly against real pixel
data: a sample point expected to be hidden shows a large color difference
from its own Cycles-only render; a sample point expected to stay visible
matches its Cycles-only color exactly (`|final - beauty| == 0.0000`).
Result: `images/32_partial_occlusion_sphere_and_volume.png` -- the sphere's
real material is clearly visible on one side and clearly replaced by the
volume's own color on the other. See `scripts/yt_idv_exploration/README.md`
("Sphere partially occluded by the volume") for the full writeup. The same
script also runs with `VOLUME_OVERRIDE_MODE=threshold_fog` (see above),
producing `images/34_partial_occlusion_fog_sphere_and_volume.png` -- a
softer, hazier version of the same partial occlusion, using the new fog
transfer function instead of MIP's harder color swap.

### Also demonstrated (retained in `scripts/yt_idv_exploration/blender_scene/`)

A real Blender scene -- flagged cube, orbiting camera, reference wireframe
boxes visualizing the sample dataset's AMR grid structure (mapped onto the
flagged object's bounding box; a placeholder mapping, not real spatial
registration), and a composited marker sphere -- driving `operators.
build_camera_params()` across multiple frames and getting back a genuinely
different yt_idv render each time. Confirms the camera-motion path works
through the real boundary, not just a single static call. See that
directory's own README for the mechanics (a subprocess bridge, needed only
because `bpy` and `yt_idv` can't share one Python in the environment this
was built in -- not a real Blender limitation) and what else it caught
(Empties don't render; use real wireframe mesh geometry -- `hide_render`,
not `display_type='WIRE'`, is what actually keeps an object out of
Workbench's rendered output -- and a large, dead-centered sphere reads as a
flat sticker, not "partly in front of" the content, until shrunk/offset).
This predates the Cycles Z-buffer wiring and still uses the OLD flat
alpha-over composite technique for its multi-frame orbit -- the demo above
is the real-depth-clip redo, but only for a single still frame.

## Architecture

### Per-frame pipeline (F12 / animation render)

1. `operators.render_current_frame()` hides flagged objects from the main
   scene (`hide_render`, not deletion) and runs **Pass 1**: a real Cycles
   render with `view_layer.use_pass_z = True`, its compositor set (by
   `compositor_build.build_pass1_capture_graph`) to do nothing but write
   Combined and Z to full-float EXR via File Output nodes.
2. Reads the Z EXR back into a `DepthBuffer` (`operators.read_depth_buffer`).
3. Builds one `RenderRequest` per flagged object (camera intrinsics +
   extrinsics, the depth buffer, that object's transform/bbox/custom_params,
   frame number) and calls `render_volume()`. Each `RenderResult.rgba` is
   written to its own full-float EXR.
4. Rebuilds the compositor (`compositor_build.build_or_update_graph`) as
   `Image(beauty EXR) -> chained Alpha Over per volume (farthest first) ->
   Composite`, then runs **Pass 2**: `scene.render.engine` temporarily
   swapped to Workbench, `write_still=True`. This is what actually produces
   the saved output file.
5. Object visibility and `scene.render.engine` are restored in a `finally`,
   whether or not any step above raised.

### Why two render passes, not one

Not explicit in the spec's lettered steps, so worth calling out the same way
as the two open questions below: Blender runs the compositor as part of the
*same* `render.render()` call that produced the 3D render, before that call
returns. There is no handler hook that fires between "this frame's Z pass
exists" and "this frame's compositor has already consumed whatever was in
its Image nodes." So a single render call cannot both produce this frame's Z
pass *and* composite this frame's volume result from it -- the volume result
depends on the Z pass, which only exists after the compositor for that same
call already ran.

Resolution: two internal render calls per frame, both triggered by our own
operator (the spec's "or an overridden render operator" option, chosen over
handlers alone for exactly this reason):

- **Pass 1** (real Cycles): produces the Z pass; its own compositor output
  is discarded, we only want the two EXRs it writes to disk.
- **Pass 2** (Workbench, essentially free): its compositor graph reads
  *everything* from EXRs via Image nodes -- no live Render Layers dependency
  -- so pass 2's actual 3D render is irrelevant and can run on a cheap
  rasterizer purely to make Blender execute the compositor and write the
  real output file.

### F12 takeover

Blender has no "override this built-in operator" mechanism. `handlers.py`
disables the default keymap item(s) binding `render.render` to F12 in the
default keyconfig, and adds `render.volume_override` on F12 (single frame)
and Ctrl+F12 (animation) in the add-on keyconfig layer instead, reversibly
(restored on unregister). The operator is also added to the Render menu for
discoverability, and callable directly as `bpy.ops.render.volume_override()`.

**Bypass caveat:** calling `bpy.ops.render.render()` directly (Python
console, another add-on) skips all of the above -- you get plain Cycles with
flagged objects hidden and no compositing, not permanently-hidden objects
(the `render_cancel` safety net in `handlers.py` guards against that one
failure mode specifically).

### Multiple flagged volumes per frame (open question 1)

Composite each flagged object independently against the same Cycles beauty
+ Z (they all see the same solid geometry, hidden the same way), then chain
them back-to-front via Alpha Over, ordered by camera distance (farthest
object's center to nearest). A nearer volume then correctly draws over a
farther one.

**Does not handle:** two volumes actually interleaved in depth along the
same ray (volume-through-volume self-occlusion) -- they still just layer in
whole-object distance-sorted order, not blend physically correctly pixel by
pixel. Out of scope for v1, same category as the solid-fully-enclosed-by-
volume limitation below.

### Camera representation (open question 2)

Pass raw camera intrinsics/extrinsics (as specified in `CameraParams`) and
let the renderer derive rays itself, rather than precomputing a per-pixel
ray-origin/ray-direction grid on the Blender side. Reasons:

- The renderer almost certainly needs to do this derivation internally
  anyway for its own volumetric marching, so precomputing it here doesn't
  remove real work, just relocates it.
- A resolution × 2 (or ×6, for origin+direction) float array is a lot of
  data to carry across a transport boundary we've deliberately kept
  swappable (in-process now, maybe subprocess/network later); raw
  intrinsics are a few floats.
- If this turns out wrong once the real renderer is wired in, `CameraParams`
  is unchanged either way -- a `ray_grid` field can be added alongside it
  without renegotiating the rest of the contract (per the spec's own
  instruction), it just hasn't been needed yet.

## Camera projection / invertibility

If your renderer wants a clean `pixel ↔ ray/point` mapping rather than
hand-deriving one from the raw intrinsics above, `CameraParams` carries two
additional, purpose-built fields for exactly that -- added alongside the raw
intrinsics per the same "don't replace, add alongside" rule as the
ray-grid option above:

- **`projection_matrix`** (PERSP/ORTHO): a 4×4 camera-to-clip-space matrix
  from Blender's own `Object.calc_matrix_camera()`. Composed with
  `matrix_world.inverted()` it's one ordinary invertible homogeneous
  transform: `world point → pixel` is the forward direction; `pixel + depth
  → unique world point` is the inverse (the depth you already get from
  `DepthBuffer` resolves the usual pixel→ray depth ambiguity into an exact
  point). **Empirically round-trip verified**, not just derived from the
  formula: `scripts/verify_camera_projection.py` places a marker at a known
  3D point, renders it with real Cycles, and checks the marker's actual
  rendered pixel centroid against this matrix's predicted pixel -- both
  PERSP and ORTHO matched to sub-pixel accuracy (≤0.55px). `None` for PANO,
  which has no single-matrix representation.
- **`panorama`** (`PanoramicLensParams`, PANO only, else `None`): the raw
  Cycles panorama fields (`panorama_type`, fisheye lens/fov/polynomial
  coefficients, latitude/longitude range) passed through verbatim rather
  than pre-converted, so you implement Cycles' own per-sub-type ray formula
  directly and match it exactly. For `EQUIRECTANGULAR`, **empirically
  confirmed** (`scripts/verify_pano_equirect.py`, same marker technique --
  six markers placed along the camera's local ±X/±Y/±Z axes, rendered, pixel
  location measured) camera-local formula (forward = -Z, up = +Y, right =
  +X):

  ```
  theta = atan2(x, -z)                     # longitude
  u = (theta / (2*pi) + 0.5) % 1.0          # 0.5=forward, 0.75=right, 0.25=left, 0/1=back
  phi = asin(y / length((x, y, z)))         # latitude
  v_from_top = 0.5 - phi / pi               # 0=up(+Y), 1=down(-Y), 0.5=equator
  ```

  cleanly invertible (standard spherical/equirectangular map, singular only
  at the poles, same as ordinary lat/long coordinates):

  ```
  theta = (u - 0.5) * 2*pi
  phi = (0.5 - v_from_top) * pi
  direction = (sin(theta)*cos(phi), sin(phi), -cos(theta)*cos(phi))
  ```

  Measured error was small enough to be confident (forward/right/left within
  ~0.5% of the predicted `u`; the "back" marker straddles the equirect seam
  and needed a wraparound-aware circular mean instead of a naive centroid to
  measure correctly -- see the script for why, and don't reuse a plain
  linear pixel-centroid near that seam elsewhere). The other panorama
  sub-types (`FISHEYE_EQUIDISTANT`, `FISHEYE_EQUISOLID`,
  `FISHEYE_LENS_POLYNOMIAL`, `MIRRORBALL`, `CENTRAL_CYLINDRICAL`,
  `EQUIANGULAR_CUBEMAP_FACE`) are documented in `PanoramicLensParams`'s
  docstring from Cycles' known camera model but **not** independently
  verified the same way -- re-run the marker technique for whichever one you
  actually need before trusting it.
- **`dof`** (`DepthOfFieldParams`, always present): aperture/focus
  parameters (`use_dof`, `focus_distance_m`, `aperture_fstop`,
  `aperture_blades`, `aperture_ratio`, `aperture_rotation_rad`) for
  physically-based lens effects (bokeh/DOF blur) in the path tracer, not
  originally part of the contract -- added because "lens information" for
  path tracing needs more than pure pinhole geometry. `focus_distance_m` is
  resolved to a plain float on the Blender side even when DOF is driven by a
  target object (`camera.data.dof.focus_object`), as Euclidean world-space
  distance from camera to that object/bone. That Euclidean choice has **not**
  been empirically checked against Cycles' own internal DOF math the way the
  Z-pass and projection matrix were -- same category of risk as the
  Euclidean-vs-planar Z-pass question. Verify with the same marker technique
  if precise DOF matching matters to you.

**Where this lives in code**: `operators.build_camera_params()` populates
all three; `operators._panorama_settings_source()` has a version-detection
note for `panorama_type` etc. living under `camera.data.cycles` (probable,
matching this add-on's target Blender version) vs. directly on core
`Camera` (observed on the pip `bpy==5.2.0` used for verification here --
apparently merged in Blender 5.0, not independently confirmed on a real
&lt;5.0 Blender).

## Z-pass verification findings

Per the spec: don't assume the Z pass's depth convention or the pixel
buffer's row order, empirically determine them. `scripts/verify_zpass.py`
builds a minimal scene (camera at a known distance, a background plane
perpendicular to the view axis, two colored unlit-emission cards to probe
row order) and reports findings to `scripts/verify_zpass_results.json`. Run
it with:

```
uv run python scripts/verify_zpass.py
```

**Findings actually obtained** (run under pip-installable `bpy==5.2.0`,
Cycles, 64×64 render, camera at distance 10.0 from a plane perpendicular to
the view axis):

| Field | Value | Evidence |
|---|---|---|
| `depth_convention` | `"PLANAR"` | Center pixel and an off-axis pixel (still on the same flat plane) both read exactly `10.0`. A Euclidean/ray-distance convention would have read `~10.205` off-axis (computed from the camera's actual FOV) -- it did not. **This contradicts the commonly-repeated claim that Blender's Z pass is ray distance; trust the measurement, not that claim.** |
| `no_hit_value` | `1e10` exactly | Sampled a frame-corner pixel deliberately outside a background plane sized to miss the frustum corners; matches Blender's long-documented sentinel. |
| `row_order` | `"BOTTOM_LEFT_ORIGIN"`, confirmed via the add-on's own real read path | A card at world +Y (the camera's "up", given identity rotation) rendered pure red; it dominates the array's *second*-half rows (`row_order_evidence` in `verify_zpass_results.json`) read back via `bpy.data.images.load(path).pixels` -- matches `operators.py`'s `DEPTH_ROW_ORDER` assumption. |
| `units` | scene `unit_settings.scale_length` (`1.0` in this test, `METRIC`) | Raw Z values are in Blender units; multiply by `scale_length` for meters. |

### Verification caveats

**Closed this session, previously open.** `row_order` used to only be
verifiable via a `bpy.data.images.load()`-bypassing `OpenEXR`-library
fallback, because the primary path -- the add-on's own real read path,
`operators.read_depth_buffer`'s `bpy.data.images.load(path).pixels` --
reproducibly returned a 0×0/0-channel image on the EXRs this repo's own
File Output helpers wrote, misdiagnosed at the time as "a broken EXR decoder
specific to the pip `bpy==5.2.0` wheel."

That diagnosis was wrong. The actual cause, found while porting the real
add-on to Blender 5.0+ (see "Blender 5.0+ compositor" below): those helpers
wrote each EXR with a **named** `file_output_items` entry, which Blender
5.0+ writes as a named EXR "part" -- and `bpy.data.images.load()` doesn't
reliably decode a named-part EXR, on the pip wheel *or* a real Blender 5.x
install (independently reproduced against both). `compositor_build.py`'s
`_configure_single_file_output` and all three `scripts/verify_*.py` helpers
now use an empty item name, which keeps the part unnamed with plain `RGBA`
channels -- the shape `bpy.data.images.load()` does handle. Re-running
`scripts/verify_zpass.py` after that fix: `readback_method_depth`/
`readback_method_beauty` now both say `"bpy.data.images.load"` (the
fallback never triggers), and the resulting `row_order` above is fully
authoritative -- not just consistent with the add-on's assumption, but
produced by the add-on's own code path. `verify_camera_projection.py` and
`verify_pano_equirect.py` show the same primary-path success after the same
fix (their own results JSON files).

**`depth_convention` and `no_hit_value` came from real Cycles output either
way** (both read-back methods always returned the same numeric values; only
array *indexing/orientation* was ever in question), so those two findings
never carried this caveat.

## Blender 5.0+ compositor

`compositor_build.py` now supports BOTH compositor scripting APIs, detected
at runtime via `hasattr`/capability checks -- never a parsed version string
-- so the same code path handles Blender 3.x/4.x and 5.0+ transparently:

| Classic (3.x/4.x) | Blender 5.0+ |
|---|---|
| `scene.node_tree` (auto-created by `scene.use_nodes = True`) | `scene.compositing_node_group` (must create + assign a `CompositorNodeTree` node-group datablock explicitly -- `ensure_compositor_tree()` now does this) |
| `CompositorNodeOutputFile.base_path` / `.file_slots[i].path` (frame number auto-appended) | `.directory` / `.file_name` (node-level filename **template** -- a literal `####` run gets the zero-padded frame number substituted; NOT automatic, confirmed empirically) + one `.file_output_items.new('RGBA', name)` entry, with per-item `.format`/`.override_node_format = True` for plain single-layer OPEN_EXR (node-level `.format.file_format` is locked to `'OPEN_EXR_MULTILAYER'` once items exist -- `item.name` is only the in-container part/layer label, NOT the filename) |
| `CompositorNodeComposite` node type | **does not exist**. Final output is whatever's wired into a `NodeTreeInterface` OUTPUT socket (named `"Image"` by this add-on) via a `NodeGroupOutput` node -- confirmed empirically by rendering through a differently-named output socket and observing it drive the actual saved pixels |
| `CompositorNodeAlphaOver.use_premultiply` (assumed name, not independently confirmed on a real <5.0 Blender) | a `"Straight Alpha"` **input socket** instead of a node property (confirmed default value is already `False`, i.e. "treat inputs as already premultiplied" -- the behavior this add-on wants) |
| Image node reading a plain file-slot EXR back always exposes generic `"Image"`/`"Alpha"` outputs | An Image node reading back an EXR whose part *was* named (true of this add-on's own beauty/depth EXRs, via `file_output_items`) exposes its single color output **under that part's name instead** -- `_color_output_socket()` falls back to `outputs[0]` when `"Image"` isn't present, to cover both shapes without needing to know the exact part name |
| `img.colorspace_settings.name = "Linear"` always valid | that literal name **no longer exists** in Blender 5.0's default OCIO config (raises `TypeError`; the equivalent space is `"Linear Rec.709"` there) -- `compositor_build.set_linear_colorspace()` tries a short candidate list, ending in the portable OCIO `scene_linear` role alias, which should keep working even against a still-different future config |

All of the above confirmed empirically against pip `bpy==5.2.0` -- including
a real render through the full two-pass pipeline shape (`tests/
test_compositor_build.py`'s `test_build_or_update_graph_composites_named_part_beauty_and_result_exrs`)
-- not assumed from documentation. `scripts/verify_zpass.py`,
`verify_camera_projection.py`, and `verify_pano_equirect.py` independently
reached the same `directory`/`file_output_items`/`override_node_format`
shape for their own (simpler, no-final-output-needed) use of File Output
nodes, which is a second confirmation of that part of the table.

**Not yet independently verified**: all of this against a real Blender
5.0+ install (only the pip module has been available so far) and against a
real Blender < 5.0 (to confirm the classic branch still matches -- untouched
by this port, but also not re-confirmed). If either turns out to disagree,
the relevant `hasattr`-gated branch in `compositor_build.py` is the one
place to fix -- both branches are exercised independently by
`tests/test_compositor_build.py`.

## Volume properties -> `VolumeObjectParams.custom_params`

`properties.VolumeOverrideProperties` (on `obj.volume_override`) holds each
volume's look, saved in the .blend so a GUI F12 and a batch render of the
same file match:

- **Renderer** (`renderer_mode`): a `YT_IDV_*` mode, or `SAMPLE_CONSTANT`
  for pipeline tests without yt_idv.
- The current mode's settings: backend (EGL/OSMesa), colormap (Maximum
  Intensity, Projection), isocontour value range/layers/colormap, threshold
  fog range/colormap/alpha max/ramp power, MIP-alpha colormap/max/ramp
  power, and the yt transfer function with its ray length and sample factor.
  Defaults match `yt_idv_renderer.py`'s module constants.
- **Rescale Colormap Every Render**, or else the **remembered colormap
  range**: the first render with nothing remembered fits the range and
  stores it on the volume; later renders reuse it. Changing the renderer
  mode, or unchecking it, makes the next render fit a new one.
- **Alpha Source: Custom Curve** / **Color Source: Image** for Threshold Fog
  and MIP Alpha. The curve lives on a `ShaderNodeFloatCurve` in a node group
  the volume points to (a `CurveMapping` can't be a property itself); it is
  sampled into `yt_idv_alpha_curve` and replaces the ramp-power shape. The
  image is averaged across rows and resampled into `yt_idv_color_image_rgba`;
  its alpha is ignored.

`to_custom_params()` puts all of these into `custom_params` (`renderer_mode`
plus `yt_idv_*` keys), so any look change changes `params_version` and
re-renders. A free-form key/value table (`custom_params_extra`) adds
anything else, e.g. `yt_idv_field`. `yt_idv_renderer.py` stays `bpy`-free:
it only reads `custom_params`.

The add-on preferences hold only the working directory and verbose logging.

## Dataset / change tracking

The end goal here is arbitrary **scientific datasets** (a yt-loadable
simulation output, say), not just VDB -- so this is designed generic-first,
with Blender's native VDB `Volume` object support as one optional
enrichment path, not the primary one.

### Two independent things cross the boundary per object, not one

1. **Shading knobs** (`VolumeObjectParams.custom_params` -- density scale,
   color, any `custom_params_extra` row): cheap to recompute and re-apply
   every frame.
2. **The dataset itself** (`VolumeObjectParams.dataset`, a `DatasetParams`):
   potentially large, potentially expensive to load/parse -- the thing you
   actually want to avoid redoing unless it truly changed.

Each gets its own `VersionInfo` (a content hash + a convenience `changed`
bool): `VolumeObjectParams.params_version` for (1), `DatasetParams.version`
for (2). Splitting them means a camera-only move across an animation (which
changes nothing about the dataset) doesn't look like "everything changed" to
a renderer trying to decide what's worth re-loading.

### How to actually use this (renderer side)

**The hash is the real cache key, `changed` is a convenience shortcut, not a
substitute.** Keep your own cache keyed by `(DatasetParams.dataset_id, DatasetParams.version.hash)`
(and separately by `(dataset_id, params_version.hash)` for shading state, if
you cache that too). When a request comes in, compare its hash to your
cache; reload/reparse only on a miss. This is correct regardless of what
Blender's `changed` flag says, and survives your process restarting,
switching transports (in-process -> subprocess -> network), or anything else
on the Blender side going sideways.

`changed=False` is a reliable "definitely nothing to do" fast path, so it's
fine to skip the hash comparison in that specific case as an optimization.
`changed=True` is NOT proof you need to do anything -- it just means Blender
doesn't remember sending this exact value before (which is trivially true
after every Blender restart or add-on reload, whether or not your renderer's
own cache is still warm). Always let the hash decide, never the boolean
alone, if you actually care about correctness across restarts.

`change_tracking.reset()` runs on every add-on registration (Blender
startup, add-on reload) so a fresh instance doesn't inherit stale bookkeeping
from a previous one. There's also a manual **Reset Change Tracking** button
(object panel and add-on preferences) for the reverse case: you know your
renderer's own cache was cleared independently and want Blender's `changed`
flag to stop reporting `False` for things that would actually need reloading.

### What actually populates `DatasetParams`

`operators.build_dataset_params()`:

- **Primary path, any object**: `properties.py`'s `dataset_path` (a path/URI
  to whatever your loader expects) and `dataset_kind` (a free-form label --
  `"yt"`, `"hdf5"`, `"vdb"`, anything; Blender never interprets it, it's
  pure passthrough for the renderer to dispatch on). Any loader-specific
  argument beyond a bare path (a yt field name, an HDF5 dataset key, a unit
  system override) goes in `custom_params_extra`, same free-form table as
  everything else -- there's no dataset-format-specific typed field here on
  purpose, since the whole point is not being tied to one format.
- **Bonus path, `obj.type == "VOLUME"`**: if the flagged object happens to
  be a Blender-native `Volume` datablock (VDB support), `dataset_path`/
  `dataset_kind` are overridden from what Blender already knows
  (`obj.data.filepath`, defaulting `dataset_kind` to `"vdb"`), and the
  sequence fields (`is_blender_volume_sequence`, `sequence_frame_start/
  duration/offset`, `sequence_mode`, `resolved_sequence_frame`, `grid_names`)
  get populated too. `resolved_sequence_frame` replicates Blender's
  documented start/duration/offset + clip/extend/repeat/ping-pong semantics
  in plain Python (`operators.resolve_sequence_frame()`) -- **not**
  empirically verified against a real VDB sequence the way the Z-pass/camera
  findings were; re-check against actual sequence data before trusting the
  exact frame number.
- **`dataset_id`**: `properties.ensure_dataset_uuid(obj)`, a UUID generated
  once and stored in the .blend file -- NOT `obj.name`, so renaming the
  object doesn't look like a brand-new dataset with no history to whatever's
  caching against this id (Blender's own `change_tracking.py` bookkeeping
  included).

## No lighting/shadow interchange (v1, by design)

Neither direction of the `RenderRequest`/`RenderResult` contract carries
light lists, shadow maps, or irradiance data -- the renderer is fully
self-emissive per the spec. **Future extension point:** if Cycles-cast
shadows/bounce light on the volume are wanted later, the natural shape is a
per-light shadow-map-style depth pass added alongside `DepthBuffer`, not a
replacement for it.

## Known limitations (v1, all by explicit design)

- **Solid fully enclosed by volume on both near and far sides of the same
  ray**: not handled. A single per-pixel Z bound can only clip integration
  at *one* depth; a solid poking into the *middle* of a volume along a ray
  needs multiple depth intervals (deep compositing) to get right. See test
  scene (d) below for how to see this limitation, not just read about it.
- **Volume-through-volume occlusion** between multiple flagged objects:
  whole-object back-to-front layering only (see open question 1 above).
- **No Z-Combine anywhere**, deliberately -- a single per-pixel depth isn't
  physically meaningful for a volumetric pass the way it is for two solids;
  its hard switch would produce artifacts wherever a solid partially
  occludes/interpenetrates the volume. Depth-clipped integration bound +
  premultiplied Alpha Over instead.
- **Viewport/Eevee preview**: out of scope, F12/animation-render only.
- **Motion blur matching**: out of scope.

## Test scenes to verify depth-clipped integration

Build these once the real `render_volume()` is wired in (or with the debug
fake renderer enabled, to confirm the *pipeline* mechanics independent of
the real renderer's correctness):

- **(a) Volume behind a solid mesh**: a cube between the camera and the
  flagged volume object. Expect the cube to fully occlude the volume behind
  it -- the volume's integration should stop at the cube's Z, contributing
  nothing to those pixels.
- **(b) Volume in front of a solid mesh**: reverse of (a) -- the mesh should
  be fully hidden by the volume in those pixels, volume renders normally
  (its far bound is the mesh's Z, but the mesh is farther than where the
  volume would matter anyway).
- **(c) A solid partially poking into the volume from one side**: e.g. a
  cube whose near face is inside the volume's bounding box but whose far
  face is beyond it. Expect a clean depth-clipped boundary along the cube's
  near surface -- volume renders normally where unoccluded, stops exactly at
  the cube where it pokes in. This is the case the Z-Combine approach would
  get wrong (hard per-pixel switch instead of a soft integration-bound clip)
  and is the main thing to actually stare at the render of.
- **(d) A solid fully enclosed by the volume** (near and far sides both
  inside the volume's extent along the same ray): expect this to render
  *wrong* -- the single Z bound clips integration at the solid's *near*
  face only, so the volume behind the solid keeps contributing to those
  pixels as if the solid weren't there past it, instead of correctly having
  two separate near/far volume intervals around the solid. This is the
  explicitly out-of-scope limitation from the spec -- (d)'s purpose is to
  make it *visible* for future work, not to fix it now.

## Development & testing

This repo uses `uv` for a **dev-only** Python environment -- Blender never
reads `pyproject.toml`; it's purely for running tests and the verification
script without needing a full Blender install:

```
uv sync                                            # installs numpy, pytest, bpy==5.2.0, OpenEXR
uv run pytest                                      # unit tests (properties, compositor
                                                    #   node-graph helpers, request
                                                    #   building/sorting, camera params)
uv run python scripts/verify_zpass.py              # empirical Z-pass check
uv run python scripts/verify_camera_projection.py  # empirical PERSP/ORTHO projection check
uv run python scripts/verify_pano_equirect.py      # empirical equirectangular mapping check
```

(All three verification scripts touch a real Cycles render, not just Blender's Python API, so `LD_PRELOAD` from your shell profile may need unsetting if it references something not present in a minimal environment: `env -u LD_PRELOAD uv run python scripts/....py`.)

### Headless yt_idv testing, in this same venv

`uv sync --group ytidv` installs real `yt_idv` (from git -- see "Installing
yt_idv into your Blender's Python" above for why) plus `pooch`/`pandas`/
`h5py` alongside everything `uv sync` already installs, into this SAME
Python-3.13 venv as `bpy`. Confirmed empirically: `bpy` and `yt_idv` import
and run together in one interpreter (a real offscreen EGL context creates
and renders fine), so `scripts/yt_idv_exploration/blender_scene/
render_real_demo.py` -- the real end-to-end demo (see "The real end-to-end
demo" below) -- now runs entirely through `uv run --group ytidv python
...`, no separate throwaway venv or subprocess bridge needed:

```
env -u LD_PRELOAD uv sync --group ytidv
env -u LD_PRELOAD uv run --group ytidv python \
    scripts/yt_idv_exploration/blender_scene/render_real_demo.py
```

This used to require a second, throwaway Python-3.12 venv
(`scripts/yt_idv_exploration/setup_ytidv_venv.sh`) plus a subprocess bridge
(`blender_scene/ytidv_worker.py`) purely to work around `yt_idv`'s
now-fixed unconditional `imgui` dependency (see "Installing yt_idv into
your Blender's Python" above) -- those scripts are kept for the historical
record of that exploration (per this repo's own "retained, not just
described" habit) but are no longer the recommended path; prefer the
`ytidv` group above for anything new. One separate, real incompatibility
surfaced while retiring the bridge: Blender's own realtime compositor grabs
a GPU/EGL context that collides with yt_idv's own offscreen EGL context in
the same process -- `operators.render_current_frame` now works around this
itself (forces `scene.render.compositor_device = "CPU"` for yt_idv-mode
renders specifically) rather than something you need to handle by hand; see
`yt_idv_renderer.py`'s module docstring for the full finding.

`bpy==5.2.0` is the pip-installable "Blender as a Python module" package
(no GUI, headless), used only for tests/verification -- it is a genuinely
different Blender (5.2) than whatever real Blender loads this add-on at
runtime. `compositor_build.py` now detects and supports both the classic
(3.x/4.x) and 5.0+ compositor APIs at runtime (see "Blender 5.0+
compositor"), so this dev dependency's version doesn't need to match your
real Blender's for the tests to be meaningful -- only the >=5.0 branch gets
exercised by this particular `bpy` wheel, though. `requires-python` is
pinned to `>=3.13,<3.14` only because that's what this `bpy` wheel needs; it
does not constrain the add-on's actual runtime Python (whatever your real
Blender embeds).

If/when you point me at your real Blender's Python, the more useful move is
to run `scripts/verify_zpass.py` there directly and diff its
`verify_zpass_results.json` against the one already committed here, plus
smoke-test `render.volume_override` (F12) end to end against a real
`render_volume()` implementation and the test scenes above.
