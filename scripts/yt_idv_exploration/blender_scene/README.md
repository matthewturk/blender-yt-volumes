# Blender-side yt_idv exploration: camera motion + AMR grid reference boxes

Answers two questions asked directly: does the real add-on's camera-building
code (`operators.build_camera_params()`) interact sanely with the
yt_idv stub across a MOVING Blender camera, and what does inserting
reference objects for the dataset's structure into the Blender scene look
like?

**This directory's subprocess bridge is now historical** -- it existed to
work around yt_idv's now-fixed unconditional `imgui` dependency (see
`../README.md` and top-level README "yt_idv integration"). `bpy` and real
`yt_idv` now coexist in one Python-3.13 venv (`uv sync --group ytidv`);
`render_real_demo.py` (one level up) was ported off the bridge as the
worked example -- prefer that pattern for anything new. The rest of this
README describes the original bridge-based exploration for the record.

## Why a subprocess bridge (historical)

`bpy` (this repo's own dev venv) and `yt_idv` (the throwaway venv, see
`../setup_ytidv_venv.sh`) couldn't be installed together here before the
imgui fix above -- same root cause as everywhere else in this exploration
(see `../README.md`). In real Blender this wasn't an issue
(`yt_idv_renderer.py` calls yt_idv in-process, see its module docstring) --
`ytidv_worker.py` existed purely so this sandbox could drive the real
yt_idv render function with a REAL Blender camera.

`render_frames_in_blender.py` (run with this repo's own venv, has `bpy`)
builds the scene and drives the camera; `ytidv_worker.py` (run in the
yt_idv venv, invoked as a subprocess) does the actual rendering. All
`render_many` calls happen in ONE worker process invocation, not one per
frame, so the dataset-load + GL-context-creation cost is paid once, the way
it would be in a real, long-running Blender session -- see
`ytidv_worker.cmd_render_many`'s docstring.

## Running it

```
env -u LD_PRELOAD uv run python \
    scripts/yt_idv_exploration/blender_scene/render_frames_in_blender.py \
    /path/to/ytidv-venv/bin/python
```

Output lands in `output/` (already retained/committed here):
- `blender_view_NNNN.png` -- Blender's own Workbench render of the full
  scene (camera + grid-box wireframes + marker sphere) for each frame.
- `frame_NNNN.png` -- the corresponding real yt_idv MIP render, built from
  `operators.build_camera_params()`'s output for that same frame's camera.
- `sphere_only_NNNN.png` -- just the marker sphere, transparent background.
- **`composite_NNNN.png`** -- the sphere alpha-over'd onto the yt_idv frame
  -- **this is the one to look at** for an unmistakable, at-a-glance
  confirmation that real yt_idv content is there and lines up with the
  Blender camera: the bold green sphere sits squarely on top of the yt_idv
  blob in every frame, not floating off to one side or flipped.
- `scene.blend` -- the actual Blender scene, orbit camera included, openable
  directly.
- `grids.json` / `requests.json` -- the raw data exchanged across the
  subprocess bridge.

## What this confirmed

- **Camera motion works end to end through the real boundary.** Each
  frame's camera comes from `operators.build_camera_params(scene, cam_obj)`
  -- the add-on's actual function, not a hand-rolled stand-in -- serialized
  across the subprocess bridge, and produces a genuinely different yt_idv
  render each time (quantitatively confirmed, not just "looks different":
  max per-pixel diff of 220-232/255 between frames in an earlier orbit
  test).
- **Grid-box reference objects need to be real mesh geometry, not Empties.**
  First attempt used `bpy.data.objects.new(name, None)` Empties with
  `empty_display_type='CUBE'` -- these are viewport-only gizmos with no
  actual geometry and never appear in a render (Cycles or Workbench,
  doesn't matter). Fixed by using real cube meshes with a Wireframe
  modifier instead (`add_grid_box_wireframes`).
- **`hide_render`, not `display_type='WIRE'`, is what actually keeps the
  flagged cube from occluding the grid boxes in the rendered output.**
  Tried `display_type='WIRE'` first on the assumption Workbench honors it
  in `bpy.ops.render.render()` output (not just the viewport) -- confirmed
  empirically that it does NOT, in this Blender version; the cube still
  rendered fully solid. `hide_render = True` does work, and happens to
  match the real add-on's own semantics anyway (a flagged object gets
  hidden from the main render pass -- see `scene_sync.
  hidden_from_main_render`).
- **The AMR grid-box visualization looks like a real refinement
  hierarchy** (`output/blender_view_0002.png`): progressively smaller,
  nested boxes clustering toward the center, which is exactly the shape
  you'd expect from AMR refinement concentrating resolution on a region of
  interest -- even though the specific mapping onto the flagged cube's
  bounding box is an arbitrary placeholder (see
  `render_frames_in_blender.py`'s module docstring), not a real spatial
  registration.
- **The grid-box wireframes turned out too subtle to confirm anything at a
  glance** -- thin lines against a plain background don't read as
  obviously "yt_idv is really here." Added a bold, solid marker sphere
  instead (`add_marker_sphere`/`position_sphere_toward_camera`), composited
  directly onto each yt_idv frame (`composite_NNNN.png`) via a plain
  alpha-over -- unmistakable, and confirms pixel alignment between the two
  renders in the same step (a misalignment/row-flip bug would show the
  sphere floating off-center or the whole thing looking wrong; it doesn't).
- **Row order matters even for this illustrative composite**, separately
  from the production add-on's own `BOTTOM_LEFT_ORIGIN` convention
  (`operators.DEPTH_ROW_ORDER`): these PNGs go through `yt.write_bitmap()` /
  `bpy.data.images.load()`, both ordinary top-down PNG readers/writers, so
  requesting `BOTTOM_LEFT_ORIGIN` from `yt_idv_max_intensity_renderer` here
  would vertically flip the yt_idv frame relative to Blender's own PNG.
  Fixed by requesting `TOP_LEFT_ORIGIN` instead for this demo specifically
  (see `ROW_ORDER_FOR_THIS_DEMO`) -- a demo-script choice, not a change to
  what the production pipeline requests.
- **A large, dead-centered sphere reads as a flat sticker, not "partly in
  front of" the content.** First attempt (`SPHERE_RADIUS=0.7`, centered)
  nearly matched the cube's whole projected width and swallowed the entire
  visible blob -- nothing recognizable as "gas" was left showing anywhere,
  so the composite just looked like a circle pasted over a red hexagon.
  Fixed by shrinking the sphere (`SPHERE_RADIUS=0.35`) and offsetting it
  along the camera's own local +X axis (`SPHERE_SIDE_OFFSET`, recomputed
  every frame so the offset direction stays "to the right of center" from
  the camera's own point of view as it orbits) -- see `composite_0001.png`:
  the blob now visibly peeks out above/beside the sphere in every frame.
- **This composite CANNOT be made properly depth-aware with MIP output --
  not a bug, a fundamental limit of what Maximum Intensity Projection
  returns.** Raised directly: shouldn't gas genuinely in FRONT of the
  sphere (between it and the camera) stay visible, with only gas BEHIND it
  correctly hidden, rather than a flat either/or cutout? Yes, in principle
  -- but MIP only ever returns "the highest density value encountered
  along this ray", discarding entirely at WHAT DEPTH along the ray that
  peak occurred. There is no per-pixel depth to compare against the
  sphere's own depth with, for this composite OR for the production
  add-on's `yt_idv_max_intensity_renderer` (which is exactly why that
  function doesn't attempt any depth-clipping either -- see its module
  docstring). Getting real depth-aware compositing here would need one of:
  (a) a yt_idv render mode that tracks/returns per-pixel depth-of-peak
  alongside the intensity (not `max_intensity`'s current output), or
  (b) real ray-marching with transmittance accumulation where a solid
  occluder's depth can be inserted mid-integration and the colormap is
  applied to the accumulated result AFTERWARD, not baked into each sample
  before compositing -- much closer to what a real transfer-function
  renderer would need to do anyway, and the natural next step after this
  exploration, not a quick fix to bolt onto MIP mode.
