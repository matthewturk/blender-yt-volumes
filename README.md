# yt-volumes

A Blender add-on that renders flagged objects with
[yt_idv](https://github.com/yt-project/yt_idv), the GPU volume renderer for
[yt](https://yt-project.org) datasets, inside a Cycles or EEVEE render. The
volume is depth-clipped against the render's Z pass and composited back in on
F12 or animation render, so Blender geometry and simulation data share one
camera and occlude each other correctly. It handles deeply nested AMR data past
float32's precision, spherical data, time series, and yt transfer functions,
interactively in Blender or in batch from a config file. A companion Blender
extension, Deep Zoom (`extensions/deep_zoom`), authors camera paths through
deep hierarchies.

**NOTE**: Much of this was written with Claude!  That being said, the
underlying yt and yt_idv are generally (but not totally!) AI-free.

**Platforms:** Blender 5.2 or later on Linux x64, with a GPU driver that
provides EGL (NVIDIA, AMD or Intel). Windows is untested; macOS has no EGL, so
yt_idv's offscreen rendering won't work there.

## Installing

Download `yt_volumes-<version>-linux_x64.zip` (and, for authoring deep camera
paths, `deep_zoom-<version>.zip`) from the
[releases page](https://github.com/matthewturk/blender-yt-volumes/releases),
then in Blender use Edit > Preferences > Get Extensions > Install from Disk.
yt, yt_idv and their dependencies are bundled; Blender's own numpy is used.

[docs/installing-in-blender.md](docs/installing-in-blender.md) covers the
other routes: installing yt and yt_idv into a venv or Blender's own Python
and using the add-on from this checkout, or building the extension zips
yourself. Installed as an extension, the module is
`bl_ext.user_default.yt_volumes`; don't also enable a classic install of it.

The add-on's preferences hold only the **Working Directory** (per-frame EXR
intermediates; default: the system temp directory) and **Verbose Logging**.

## Quick start in Blender

1. Delete the default cube: it fills exactly the box the volume will occupy,
   so it hides the volume completely. Then in the 3D viewport, **Add > yt
   Volume**. That makes a flagged Empty-Cube with a Geometry Nodes transfer
   function on it.
2. In Object Properties > **Volume Override**, set the **Dataset Path** to
   anything `yt.load` reads, or to a yt_idv scene (`.zip`). With no path it
   renders yt's IsolatedGalaxy sample, which needs Blender's Allow Online
   Access on for the first download.
3. Press **F12**. The add-on takes over F12 and Ctrl+F12, renders the scene,
   renders the volume with yt_idv from the same camera, clipped by the
   scene's geometry, and composites the two.

The **Renderer** in that panel picks how the volume looks:

- **Maximum Intensity Projection**, **Maximum Intensity, Value-Driven
  Alpha** (opacity rising with the value) and **Projective Integration**
  (line-of-sight integration), each with a colormap;
- **Geometry Nodes Transfer Function**, **yt Layers** and **yt Transfer
  Function**, which color and fade the data by value: see
  [docs/transfer-functions.md](docs/transfer-functions.md);
- **Debug: Depth x Constant** (`SAMPLE_CONSTANT`), which needs no yt_idv.

The look is saved on the object in the .blend.

Any flagged Empty-Cube works the same way (an Empty-Sphere for spherical
data). The field defaults to density; set another with the extra custom param
`yt_idv_field` (e.g. `gas, temperature`).

## Volumes in the viewport
**Show in Viewport** (Volume Override panel, yt_idv modes) draws that volume
over each 3D viewport, rendered with yt_idv from the viewport's own view. It
re-renders a moment after the view, the frame or the scene (a volume's look,
its transfer-function nodes, geometry) changes, at **Viewport Resolution**
times the viewport's size and at most **Viewport Samples per Cell** (default
4; F12 keeps each volume's own, 16 for the yt modes). It shares F12's loaded data, so turning it on loads the data once
(Blender is unresponsive meanwhile) and F12 after that doesn't load it again.
Before either loads data, a dialog names the dataset and asks to confirm; a
file saved with Show in Viewport on doesn't load until you turn it on again.
Batch renders (`render_volume`, `render_series`) never ask.
Scene geometry occludes the volume as in F12: the viewport's depth buffer
clips yt_idv's rays. Solid and Material Preview shading draw geometry into
it; a Rendered viewport may not. Camera shift and panning in camera view
are followed; orthographic views aren't supported yet.

## Exploring in a yt_idv window
**Open yt_idv Viewer** (Volume Override panel, yt_idv modes) opens yt_idv's
own window on the volume's data, with its current look, starting from the
scene camera's view. Drag to orbit and scroll to move in. Cameras pass between
the two on request: **Enter** in the window moves the scene camera to its
view, so F12 renders what the window shows, and **Send Camera to yt_idv**
(Volume Override panel, while the window is open) moves the window to the
scene camera and its current zoom, including Deep Zoom's. Syncing on request,
not continuously, keeps an animated or constrained Blender camera from
pulling the window back while you orbit.

Zooms and Deep Zoom shot space are registered as F12 registers them; as you
scroll in, the window refits its model space around the camera, so it keeps
float32 precision at any depth. Scrolling only moves the camera: the zoom
stays the volume's or Deep Zoom's. The window runs as a separate process with
Blender's Python, so it loads its own copy of the data (on the GPU too). Not
yet: changing the look from the window (yt_idv's controls need imgui, which
doesn't build for Python 3.13) and the field of view, which stays the scene
camera's.

## Cycles or EEVEE
Pass 1 (the scene's beauty and Z pass) renders with the scene's engine,
Cycles or EEVEE; the volume always comes from yt_idv, so it looks the same
either way. What can differ:

- **Requirements.** EEVEE needs Blender's **Vulkan** GPU backend
  (Preferences > System > GPU Backend, then restart; `blender -b
  --gpu-backend vulkan` in batch). Under OpenGL, EEVEE's GL context and
  yt_idv's EGL context can't share a process, and the render stops with an
  error saying so. Cycles works under either backend.
- **The geometry, not the volume.** EEVEE rasterizes: shadows, reflections,
  refraction and indirect light are approximations, so meshes, lights and
  materials look different, as in any EEVEE vs Cycles comparison. Nothing
  refracts or casts shadows onto the volume in either engine (see "Known
  limitations").
- **Depth clipping.** With perspective cameras EEVEE's Z pass matches
  Cycles' (planar depth, 1e10 where nothing is hit, values within about
  1e-5 relative), so the volume cuts into geometry in the same place. Measured
  on `examples/simple.blend` with the PopII scene: mean RGB difference
  0.06/255, with differences only along the sphere's silhouette.
- **Panoramic cameras.** EEVEE renders equirectangular panoramas, but its
  depth differs from the exact panorama depth by a median 0.4% and up to a
  few percent, and silhouettes shift by a pixel. The volume's clip can be
  visibly off near occluders, so the add-on warns; use Cycles for panoramas.
- **Settings.** `blender.samples` sets EEVEE's render samples; `device` and
  the OptiX denoiser are Cycles-only. Cycles' persistent-data cache (and the
  care taken not to invalidate it) doesn't apply to EEVEE.

## Batch rendering from a config file
`render_volume` renders a yt dataset into an existing `.blend` without a
custom harness: one TOML file names the dataset (plus optional subset), the
`.blend` file, the Empty-Cube object to replace, and the look.

```
PYTHONPATH= uv run --group ytidv python -m yt_volumes.render_volume examples/render_volume.toml
PYTHONPATH= uv run --group ytidv python -m yt_volumes.render_volume \
    --sample IsolatedGalaxy --blend scene.blend --volume "Volume 1" --frames 1-10 \
    --set data.subset.kind='"sphere"' --set data.subset.radius='[20, "kpc"]'
```

`examples/render_volume.toml` documents every key. Options override the
file, `--set section.key=value` reaches any key (values parse as TOML), and
`--check` prints the merged config without rendering. `[render]` keys
override the volume's own settings in the `.blend`; with no `[render]` the
`.blend` renders exactly as authored in the GUI. Each run gets its own
temporary `work_dir` unless one is set, so concurrent runs don't collide.
`blender.engine = "BLENDER_EEVEE"` renders with EEVEE (see "Cycles or EEVEE";
run Blender with `-b --gpu-backend vulkan`).

## Animating a dataset series
`render_series` drives `render_volume` over a time series. The config is the
same, except `[data]` names a `series` (glob or list of paths) or a
`simulation` (`yt.load_simulation` parameter file and code), and `[time]`
maps frames to outputs: by simulation time or by output index, with optional
`[frame, time]` keyframes to stretch or hold. `examples/render_series.toml`
has the details.

```
PYTHONPATH= uv run --group ytidv python -m yt_volumes.render_series examples/render_series.toml --dry-run
```

Consecutive frames that show the same output form a segment, and each segment
runs as its own `render_volume` process, so a crash loses only that segment
and each dataset's memory is returned when its segment ends. Inside Blender
(`blender -b --python-expr "from yt_volumes.render_series import main; main()"
-- series.toml`) segments run as `blender -b`, inheriting `--gpu-backend` and
`--online-mode`/`--offline-mode`. Each segment's resolved config is saved under
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

## Rendering a subset of a dataset
A volume provider can pass any yt data object to `add_scene` -- `ds.region`,
`ds.sphere`, `ds.cut_region` -- and the Blender volume box still maps to
yt_idv's unitary `[0, 1]^3`. Unscaled, the subset sits at its true position and
size inside that cube, so it looks small. `yt_idv_renderer.
scale_scene_to_subset(scene)` stretches the scene so the
subset's longest axis fills the cube, starting at its origin corner; shorter
axes leave empty space at their high end. It leaves full-domain scenes alone.
`scripts/headless/scene_setup.py`'s `SUBSET_HALF_WIDTH_KPC`/`SCALE_SUBSET` show both.

The stretch uses the data object's own bounding box; grid blocks that stick
out past it are masked off by yt_idv's selector bitmap. Projection values
also grow by the stretch factor, so fixed colormap ranges
from an unscaled run don't carry over.

## Deep zooms past float32
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

While zoomed, **Limit Zoom Depth** (on by default) ends rays at **Zoom Depth**
(default 2) times the camera's distance to the target, a little past it, so the
sparse gas filling the rest of the box behind it doesn't haze the view.

Either way the worker rescales yt_idv each frame so the camera is about one
unit from the target in model space, and maps world to model coordinates
relative to the target in float64. `examples/make_flythrough_blend.py` turns
`flythrough_densest.py`'s camera paths into either kind of scene (`--fly`).

`docs/zoom-movies.md` is a recipe for a zoom movie with yt Layers: fixed
layers, fading the shells the camera passes, and calibrating Ray Length as the
exposure.

## The Deep Zoom extension
`extensions/deep_zoom` is a self-contained Blender extension for designing
camera paths through deeply nested data. It works in shot space, a float64
anchor at the world origin plus an animatable zoom, so the camera stays at
ordinary sizes at any depth. It provides shots, a grid map of the AMR
hierarchy, re-anchor and rescale cuts, a path check that says where they're
needed, and float64 path export. See its README. This renderer picks up its
scene settings automatically; `examples/deep_zoom_shots.toml` renders a
two-shot dive into DeeplyNestedZoom authored by
`examples/make_deep_zoom_shots_blend.py`.

## Spherical data
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

## Stereoscopic renders
With Output Properties > Stereoscopy on in Stereo 3D mode, each eye is
rendered separately: the volumes are rendered from that eye's camera, clipped
against that eye's depth, and composited onto that eye's image, so the output is
whatever Stereoscopy is set to write (individual images, anaglyph,
side-by-side). The eye cameras follow the camera's Stereoscopy settings
(interocular and convergence distance, Off-Axis, Parallel or Toe-in, pivot), and
match where Cycles draws each eye to under a pixel. The second eye reuses the
first eye's Auto Exposure, so the pair matches. Each eye costs a full volume
render. Multi-View mode and panoramic stereo aren't supported.

## Scene units per data unit
Each volume keeps `obj["yt_units"]` current, for drivers, Geometry Nodes inputs
and scripts:

- `scene_per_unitary`: scene units per unitary length now, following the
  volume's transform and box, its Zoom Level, or Deep Zoom's zoom and World Scale.
- for each of the panel's **Data Units** (comma-separated yt units, e.g.
  `kpc, pc, Mpc/h, kpccm`): `unitary_in_<unit>` and `scene_per_<unit>`.

A driver reads them with a Single Property variable, e.g.
`["yt_units"]["scene_per_kpc"]`, which needs no Python auto-run. Geometry Nodes can't
read custom properties, so **Units for Geometry Nodes** (beside Data Units)
makes a one-vertex mesh, linked as the volume's Units Object, whose point
attributes hold the same values every frame: Object Info on it, then Sample
Index 0 of a Named Attribute such as `scene_per_kpc`. The data side
is yt's unit registry, recorded when the dataset loads (F12 or the viewport
preview) and saved in the .blend; **Read Units** (the button beside Data Units)
reads it from the dataset's header without loading its data. A yt_idv scene file
carries it as a `dataset` entry in scene.json (yt_idv 0.6.1 and later write
it); for scenes saved without it, `units.add_units_to_scene_file(path, ds)` appends a `dataset.json`
member without rewriting the archive.

### Placing things at unitary positions
The Units Object also carries where unitary coordinates are in the scene right
now: through the volume's transform and subset box, its Zoom Level, or Deep
Zoom. **Units for Geometry Nodes** also adds a **yt Unitary to Scene** node
group: give it the Units Object and a **Unitary** position, and its
**Position** output, in the space of the object the nodes run on, goes
straight into Set Position or Instance on Points; **Scene per Unitary** sizes
things to the data. Positions stay put as you zoom or move the volume.

Geometry Nodes works in float32, which runs out at about 1e-7 of the box. For
deeper zooms give the position in two parts, **Unitary** plus **Unitary Low**,
and the node group keeps float64 precision near the zoom target:

- typed: select the node, then the node editor's sidebar **yt** tab, **Set
  Unitary Position**, and paste `x, y, z` with all its digits;
- from Python, for catalogs: `units.store_unitary_positions(mesh, positions)`
  writes float64 `(n, 3)` positions as point attributes `unitary` and
  `unitary_low`, read with Named Attribute.

A "max" zoom target is placed once a render or the viewport preview has found
it; spherical volumes aren't supported yet.

## Examples

Each script's docstring has the commands.

| | |
|---|---|
| `examples/render_volume.toml`, `render_series.toml` | a dataset or a time series into an existing .blend |
| `examples/popii_dive.toml` + `make_deep_zoom_shots_blend.py` | a saved yt_idv scene, a two-shot dive authored with Deep Zoom, yt shells |
| `examples/deep_zoom_shots.toml`, `deep_zoom_shots_tf.toml` | the same for the DeeplyNestedZoom sample, 12 decades |
| `examples/deep_zoom.toml` + `make_deep_zoom_blend.py` | a 12-decade dive with the volume's own zoom target/level |
| `examples/spherical_earth.toml`, `earth_spin.toml` + `make_earth_blend.py` | spherical data in an Empty-Sphere; time-varying |
| `examples/make_flythrough_blend.py` | a `flythrough_densest.py` path as a Deep Zoom scene |
| `examples/shells_tf.py` | yt transfer functions that follow the camera |
| `examples/cosmology_halos.toml` + `make_cosmology_halos_blend.py` | an Enzo cosmology run in physical time with yt_astro_analysis halos as spheres (needs `yt_astro_analysis`) |

## Known limitations

- A solid fully enclosed by a volume, with volume both in front of and behind
  it along a ray, clips the volume only at its near face; the volume behind
  it still shows. Only one depth per pixel reaches yt_idv.
- Two volumes that overlap in depth are layered as whole objects, farthest
  first, not blended along each ray.
- No lighting or shadows pass between Blender and the volume: the volume is
  self-emissive, and it doesn't cast shadows or light onto geometry.
- The viewport preview doesn't support orthographic views. Motion blur
  isn't matched.
- Opaque End (yt Layers) needs a yt_idv newer than 0.6.1 (yt-project/yt_idv#312).

## Development

Needs Python 3.13 (the pip `bpy` module the tests and batch renders use is
Blender 5.2's), [uv](https://docs.astral.sh/uv/), and a GPU with EGL. Run
commands with `PYTHONPATH` unset.

```
git clone https://github.com/matthewturk/blender-yt-volumes.git && cd blender-yt-volumes
PYTHONPATH= uv sync --group ytidv       # the add-on, bpy, yt, yt_idv
PYTHONPATH= uv run --group ytidv pytest -q
PYTHONPATH= uv run --group ytidv python examples/make_simple_blend.py examples/simple.blend
PYTHONPATH= uv run --group ytidv python -m yt_volumes.render_volume \
    examples/render_volume.toml --frames 1 --set blender.device='"CUDA"' --online
```

That renders frame 1 of `examples/render_volume.toml`, a 50 kpc region of
yt's HiresIsolatedGalaxy sample, into the simple scene, to
`renders/frame_0001.png`. Set `blender.device` to `OPTIX`, `CUDA`, `HIP` or
`CPU` for your GPU. `--online` lets yt download the sample on first use.

To use this checkout in a real Blender, symlink `src/yt_volumes/` into its
add-ons directory and make yt and yt_idv importable there (see the install
guide). The pip `bpy` and your Blender are different builds; the tests
exercise only Blender 5's compositor API.

- `scripts/headless/render_headless.py` renders a `.blend` through the
  add-on and real yt_idv; edit `scene_setup.py` beside it for the dataset
  and look. `RENDER_START_FRAME`/`RENDER_END_FRAME` limit the frames.
- `scripts/verify_zpass.py`, `verify_camera_projection.py` and
  `verify_pano_equirect.py` measure Blender's Z-pass and camera conventions
  on real Cycles renders and write the `*_results.json` beside them.
- `scripts/build_extension.sh` builds the extension zips;
  `.github/workflows/release.yml` runs it for every `v*` tag.

[docs/architecture.md](docs/architecture.md) describes the two-pass pipeline
and the renderer contract.

```
src/yt_volumes/
    __init__.py            bl_info, register()/unregister()
    properties.py          per-object flag, Dataset Path, zoom target/level, UI panel
    operators.py           two-pass render orchestration, preferences, F12 takeover
    yt_idv_renderer.py     the yt_idv renderer: registration, zoom/shot space, depth clip, modes
    render_volume.py       batch rendering from a TOML config
    render_series.py       batch rendering of a time series
    external_render.py     the renderer interface contract (dataclasses)
    compositor_build.py    compositor graphs for the two passes
    viewport_preview.py    Show in Viewport
    viewer.py, viewer_main.py, viewer_link.py   the yt_idv viewer window
    transfer_function_nodes.py, tf_plot.py      the Geometry Nodes transfer function, its plot
    units.py               scene units per data unit
    scene_sync.py, change_tracking.py, handlers.py, sample_renderers.py, ...
extensions/deep_zoom/      the Deep Zoom authoring extension (see its README)
examples/                  configs and scene builders
scripts/                   verification tools, the headless driver, the extension build
tests/                     pytest suite
```

## License

GPL-3.0-or-later (see `LICENSE`), as Blender requires of add-ons; this covers
yt-volumes and Deep Zoom. The extension zips from `scripts/build_extension.sh`
also bundle yt, yt_idv and their dependencies, which keep their own
(GPL-compatible, mostly BSD) licenses, included in each wheel.
