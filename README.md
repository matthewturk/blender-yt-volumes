# cycles-volume-override

Blender add-on: objects flagged with a custom property render via an
external, self-emissive volume renderer instead of Cycles, depth-clipped
against Cycles' Z pass, and composited back in automatically on F12 /
animation render. Final + animation render only (see "Out of scope").

## Contents

```
src/cycles_volume_override/
    __init__.py          bl_info, register()/unregister()
    properties.py         PropertyGroup: flag + custom_params fields, UI panel
    scene_sync.py          hidden sync Scene, main-scene visibility toggle
    external_render.py     the frozen interface contract + stubbed render_volume()
    compositor_build.py    programmatic compositor node graph (2 graphs: pass 1 capture, pass 2 composite)
    operators.py            two-pass render orchestration, RenderRequest building
    handlers.py             render_cancel safety net, F12/Ctrl+F12 keymap takeover
scripts/verify_zpass.py               empirical Z-pass/row-order verification tool
scripts/verify_camera_projection.py    empirical PERSP/ORTHO projection_matrix round-trip check
scripts/verify_pano_equirect.py        empirical equirectangular panorama mapping check
tests/                      pytest suite for the bpy-independent-ish pieces
pyproject.toml              dev/test tooling only (uv) -- see "Development & testing"
```

## Installing the add-on in Blender

This is a classic (`bl_info`-style) add-on, not a pip package Blender
installs for itself -- Blender doesn't consult `pyproject.toml`. Either:

- Zip `src/cycles_volume_override/` and use Blender's *Install from Disk*, or
- Symlink `src/cycles_volume_override/` into your Blender's addons directory
  and enable it in Preferences > Add-ons.

Then enable **Cycles + External Volume Renderer Override** in Preferences.
Its own preferences panel has:
- **Use debug fake volume renderer** -- `render_volume()` is a stub
  (`NotImplementedError`) until you wire in the real renderer; check this to
  exercise the whole pipeline with a flat placeholder volume color instead.
- **Working Directory** -- where per-frame EXR intermediates go (default:
  system temp dir).

## Wiring in the real renderer

Everything lives behind `external_render.render_volume(request) -> result`.
Replace its body. Don't change the dataclass shapes without also updating
whatever's on the other side of that boundary -- they're the whole point of
this file. See "Swappable transport" below for why it's structured this way.

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

### Swappable transport

`render_volume()` is the *only* function anything else calls to reach the
external renderer. Whether that's an in-process Python call (now), a
subprocess, or a network call (later) is entirely inside that one function
-- nothing in `operators.py` needs to know or change.

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
| `row_order` | `"TOP_LEFT_ORIGIN"` for the *raw on-disk EXR* | A card at world +Y (the camera's "up", given identity rotation) rendered pure red; it dominates the array's first-half rows under a direct OpenEXR-library read. |
| `units` | scene `unit_settings.scale_length` (`1.0` in this test, `METRIC`) | Raw Z values are in Blender units; multiply by `scale_length` for meters. |

### Verification caveats

**`row_order` was not closed-loop-verified end to end.** The add-on's actual
read-back path (`operators.read_depth_buffer`) uses
`bpy.data.images.load(path).pixels`, which is well-documented, longstanding
Blender behavior to be **bottom-left-origin** (it also always contains 4
channels regardless of the source EXR's channel count) -- that's what
`operators.py` assumes (`DEPTH_ROW_ORDER = "BOTTOM_LEFT_ORIGIN"`). But the
pip `bpy==5.2.0` wheel used to run the verification script has a broken EXR
decoder in this environment: `bpy.data.images.load()` silently returns a
0×0 image and misreports `file_format` as `'TARGA'` for a file confirmed
valid by the `file` utility, even though `bpy.app.build_options.image_openexr`
reports `True`. `verify_zpass.py` tries the real `bpy.data.images.load()`
path first and only falls back to the `OpenEXR` PyPI package when that
fails (it did, here) -- so the `row_order` figure above describes the raw
file's on-disk scanline order via that fallback, not proof of the in-memory
`Image.pixels` convention this add-on actually relies on. **Re-run
`scripts/verify_zpass.py` against your real Blender** (point `uv run` at its
bundled Python, or just run the script inside Blender's Python console) --
if the primary `bpy.data.images.load()` path succeeds there (it should, in a
full Blender install), `verify_zpass_results.json`'s
`readback_method_depth`/`readback_method_beauty` will say so and the
`row_order` finding becomes fully authoritative for that Blender. If it
disagrees with `"BOTTOM_LEFT_ORIGIN"`, flip `DEPTH_ROW_ORDER` in
`operators.py`.

**`depth_convention` and `no_hit_value` came from real Cycles output either
way** (both read-back methods return the same numeric values; the fallback
only affects array *indexing/orientation*, not the *values* in it), so those
two findings don't carry the same caveat.

## Blender 5.0+ compositor (known gap, discovered while building this)

While building against pip `bpy==5.2.0` (the only pip-installable version
available at the time), discovered that Blender 5.0 replaced the entire
classic compositor scripting API this add-on's own idiom targets:

| Classic (3.x/4.x) | Blender 5.0+ |
|---|---|
| `scene.node_tree` (auto-created by `scene.use_nodes = True`) | `scene.compositing_node_group` (must create + assign a `CompositorNodeTree` node-group datablock explicitly) |
| `CompositorNodeOutputFile.base_path` / `.file_slots[i].path` | `.directory` / `.file_output_items.new(socket_type, name)` + `.file_name` (node-level filename; `file_output_items` are in-container layer names, not filenames) + per-item `.format`/`.override_node_format` for single-layer output (node-level `.format.file_format` is locked to `'OPEN_EXR_MULTILAYER'` once items exist) |
| `CompositorNodeComposite` node type | **does not exist** -- compositor trees became generic node groups; final output appears to go through the tree's `NodeTreeInterface`, not explored further (out of scope -- see below) |
| `CompositorNodeAlphaOver.use_premultiply` (assumed name, not independently confirmed on a real <5.0 Blender) | a `"Straight Alpha"` **input socket** instead of a node property |

`compositor_build.ensure_compositor_tree()` detects the >=5.0 shape and
raises a clear `RuntimeError` rather than silently building a broken graph
(covered by `tests/test_compositor_build.py`). **This add-on currently only
works on Blender 3.x/4.x.** Given the project's own stated node-scripting
idiom (`node_tree.nodes.new(...)`, `.links.new(...)`) matches the classic
API, this was the reasonable default target rather than chasing a brand-new,
still-evolving 5.0 redesign speculatively -- but if your real Blender turns
out to be 5.0+, `compositor_build.py` needs a follow-up patch using the
right-hand column above as a starting map.

## Custom properties -> `VolumeObjectParams.custom_params`

`properties.VolumeOverrideProperties` (on `obj.volume_override`) exposes
`density_scale`, `absorption` (RGB), `scattering` (RGB), `color` (RGB),
`vdb_filepath` as typed fields, plus a free-form key/value table
(`custom_params_extra`) for anything the real renderer needs that isn't
anticipated here. `to_custom_params()` merges both into the one dict that
becomes `VolumeObjectParams.custom_params` -- extending the real renderer's
parameter set never requires touching this add-on's Python, just adding a
row in the object's panel.

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
- **Blender 5.0+**: see dedicated section above.

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

`bpy==5.2.0` is the pip-installable "Blender as a Python module" package
(no GUI, headless), used only for tests/verification -- it is a genuinely
different Blender (5.2) than whatever real Blender loads this add-on at
runtime, which is why `compositor_build.py` deliberately targets the older
API most of the codebase assumes (see "Blender 5.0+ compositor") rather than
whatever this particular dev dependency happens to be. `requires-python` is
pinned to `>=3.13,<3.14` only because that's what this `bpy` wheel needs; it
does not constrain the add-on's actual runtime Python (whatever your real
Blender embeds).

If/when you point me at your real Blender's Python, the more useful move is
to run `scripts/verify_zpass.py` there directly and diff its
`verify_zpass_results.json` against the one already committed here, plus
smoke-test `render.volume_override` (F12) end to end against a real
`render_volume()` implementation and the test scenes above.
