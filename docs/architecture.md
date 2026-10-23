# How yt-volumes works

For people extending the add-on or writing their own renderer for it. The
code-level detail lives in the module docstrings, especially
`external_render.py` (the renderer contract), `operators.py` (the pipeline)
and `yt_idv_renderer.py` (registration, zoom, depth clip).

## Per-frame pipeline (F12 / animation render)

1. `operators.render_current_frame()` runs **pass 1**: a render with the
   scene's engine (Cycles, or EEVEE under the Vulkan backend) with
   `view_layer.use_pass_z = True`, its compositor set (by
   `compositor_build.build_pass1_capture_graph`) to do nothing but write
   Combined and Z to full-float EXR via File Output nodes. Flagged objects
   are Empties, which never render, so nothing needs hiding.
2. It reads the Z EXR back into a `DepthBuffer` (`operators.read_depth_buffer`).
3. It builds one `RenderRequest` per flagged object (camera, the depth buffer,
   the object's transform, bounding box and look, the frame) and calls the
   renderer for that object's `renderer_mode`. Each `RenderResult.rgba` goes
   into its own float Image datablock, reused every frame, which the
   compositor reads directly.
4. It rebuilds the compositor (`compositor_build.build_or_update_graph`) as
   `Image(beauty EXR) -> Alpha Over per volume (farthest first) -> output`,
   then runs **pass 2** with the same engine and `write_still=True`. The
   graph has no Render Layers node, so nothing is rendered again; this pass
   only runs the compositor and writes the output file.
5. Scene settings the pipeline forces (Z pass, persistent data, CPU
   compositor, Standard view transform) are restored in a `finally`, or once
   at the end of a `render_session`.

### Why two passes

Blender runs the compositor inside the same `render.render()` call that
produced the 3D render, and no handler fires between "this frame's Z pass
exists" and "the compositor has read its Image nodes". The volume depends on
the Z pass, so one render call can't produce both. Pass 2 keeps pass 1's
engine because switching engines in between invalidates Cycles'
persistent-data cache.

### F12 takeover

Blender has no way to override a built-in operator. `handlers.py` disables
the default keymap items binding `render.render` to F12 and Ctrl+F12, and
binds `render.volume_override` in the add-on keyconfig instead (restored on
unregister). The operator is also in the Render menu and callable as
`bpy.ops.render.volume_override()`. Calling `bpy.ops.render.render()`
directly bypasses all of this and gives a plain render without the volumes.

### Several volumes per frame

Each flagged object is rendered against the same beauty and Z, then chained
back to front with Alpha Over, ordered by the distance from the camera to
each object's center. Two volumes interleaved in depth along the same ray
layer as whole objects, not physically.

## The renderer contract

`external_render.py` defines the dataclasses that cross between Blender and a
renderer; nothing in them depends on `bpy`, so renderers are unit-testable.
A renderer is a callable `RenderRequest -> RenderResult`, registered in
`operators.RENDERER_MODES`.

- **`CameraParams`**: raw intrinsics and extrinsics (`matrix_world`, lens,
  sensor, shift, clip range, resolution), plus:
  - `projection_matrix` (PERSP/ORTHO): Blender's `calc_matrix_camera()`.
    With `matrix_world.inverted()` it maps world points to pixels, and a
    pixel plus its depth back to a unique world point.
    `scripts/verify_camera_projection.py` checks it against a real Cycles
    render to under 0.55 px.
  - `panorama` (PANO only): Cycles' panorama fields, passed through as is.
    Equirectangular is verified by `scripts/verify_pano_equirect.py`, with
    camera-local forward = -Z, up = +Y:
    ```
    theta = atan2(x, -z);  u = (theta / (2*pi) + 0.5) % 1.0
    phi = asin(y / length((x, y, z)));  v_from_top = 0.5 - phi / pi
    ```
    The other panorama types are documented from Cycles' camera model but
    not verified.
  - `dof`: aperture and focus distance. A focus object is resolved to its
    Euclidean distance from the camera, which hasn't been checked against
    Cycles' own DOF.
- **`DepthBuffer`**: the Z pass as `(resolution_y, resolution_x)` float32.
  `scripts/verify_zpass.py` measures its conventions on a real render:
  depth is **planar** (distance along the view axis, not along the ray),
  background pixels are exactly `1e10`, rows are **bottom-left origin**, and
  values are in Blender units (times `unit_settings.scale_length` for meters).
- **`VolumeObjectParams`**: the object's `matrix_world`, `bounding_box_local`,
  `custom_params` (the look) and `dataset` (`DatasetParams`).
- **`RenderResult`**: `(resolution_y, resolution_x, 4)` float32,
  **premultiplied** alpha, **linear** color, plus an optional `cmap_range`
  the add-on remembers on the volume.

Only one depth per pixel crosses the boundary, so a solid enclosed by a
volume along a ray clips the volume at its near face only; the volume behind
it still contributes. There's no Z-Combine anywhere: a single depth isn't
meaningful for a volume, and its hard per-pixel switch would show wherever a
solid cuts into one. No lights or shadows cross either; the renderer is
self-emissive.

### The look: `custom_params`

`properties.VolumeOverrideProperties` (`obj.volume_override`) holds each
volume's look, saved in the .blend so a GUI F12 and a batch render of the
same file match. `to_custom_params()` writes it into `custom_params`
(`renderer_mode` plus `yt_idv_*` keys), so any change re-renders. The
free-form `custom_params_extra` table adds anything else, e.g.
`yt_idv_field`. `yt_idv_renderer.py` reads only `custom_params`, never `bpy`.

Colormap ranges for Maximum Intensity, Value-Driven Alpha and Projection are
fitted on the first render and remembered on the volume, unless **Rescale
Colormap Every Render** is on. Changing the mode refits.

### Datasets and change tracking

The look and the dataset are versioned separately, so a camera move doesn't
look like a new dataset:

- `VolumeObjectParams.params_version` tracks `custom_params`.
- `DatasetParams.version` tracks the dataset: `dataset_path`, a free-form
  `dataset_kind` and a `dataset_id`, a UUID stored on the object so renaming
  it doesn't look like a new dataset. A Blender `Volume` object fills these
  from its VDB file and sequence settings instead.

Each `VersionInfo` has a `hash` and a `changed` flag. Cache on
`(dataset_id, hash)`: `changed=False` reliably means nothing changed, but
`changed=True` only means Blender doesn't remember sending that value, which
is true after every restart. `change_tracking.reset()` runs on registration;
**Reset Change Tracking** in the panel does it by hand.

## yt_idv registration

The flagged object's bounding box and `matrix_world` map yt_idv's unitary
`[0, 1]^3` domain into Blender world space. The same mapping positions the
yt_idv camera and turns the Z pass into yt_idv window depth, so its rays stop
at Blender geometry (`set_external_depth_clip`). Zooms magnify about a
float64 target and rescale yt_idv's model space each frame, so the GPU's
float32 stays precise at any depth.

A volume's data comes from its Dataset Path, or from a **volume provider**:
`obj.volume_override.yt_idv_volume_provider = "module:function"` names a
`provider(ctx) -> VolumeSetupResult`, called per object per frame, that
builds the yt_idv scene itself (see `yt_idv_renderer.py`).

## Compositor APIs

`compositor_build.py` supports both compositor APIs, chosen by capability
checks rather than version numbers:

| Blender 3.x/4.x | Blender 5.0+ |
|---|---|
| `scene.node_tree`, created by `scene.use_nodes = True` | `scene.compositing_node_group`, a `CompositorNodeTree` created and assigned explicitly |
| File Output `base_path` / `file_slots[i].path`, frame number appended | `directory` / `file_name` template (`####` becomes the frame) and a `file_output_items` entry with its own OPEN_EXR format |
| `CompositorNodeComposite` | No Composite node: the output is a `NodeGroupOutput` on an `"Image"` interface socket |
| Alpha Over's `use_premultiply` | a `"Straight Alpha"` input socket, default off |
| `"Linear"` colorspace | `"Linear Rec.709"`; `set_linear_colorspace()` falls back to the `scene_linear` role |

File Output items are left unnamed: Blender 5 writes a named item as a named
EXR part, which `bpy.data.images.load()` doesn't read reliably. Only the 5.0+
branch is exercised by the pip `bpy` the tests use.

## The sample renderer

`sample_renderers.constant_density_renderer()` (Renderer: `SAMPLE_CONSTANT`)
exercises the whole contract without yt_idv: it builds per-pixel rays from
`CameraParams` (PERSP, ORTHO, equirectangular), intersects them with the
volume's box, clips at the depth buffer, and outputs path length times a
constant. Use it to test pipeline changes in seconds.

## Scenes for checking depth clipping

With any renderer, including `SAMPLE_CONSTANT`:

- **A solid in front of the volume** should hide the volume behind it.
- **A solid behind the volume** should be covered by it.
- **A solid poking into the volume from one side** should cut a clean
  boundary along its near surface. This is the case a Z-Combine would get
  wrong.
- **A solid fully inside the volume** renders wrong by design: the volume
  behind it still shows (see "The renderer contract").
