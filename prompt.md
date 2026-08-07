# Task: Blender add-on for per-object render-engine override (Cycles + external volume renderer)

## Goal
Build a Blender Python add-on that lets specific objects (marked via a custom
property) be rendered by an external volume renderer instead of Cycles, while
everything else renders normally in Cycles, composited together automatically
on F12 / animation render.

I have a working volume renderer (Python-callable) that returns RGBA for a
given volume object and camera/scene parameters. It is exclusively volumetric,
self-emissive (it computes its own emission/color internally — it does NOT
need or expect scene lighting, shadow maps, or light lists from Blender).
Do NOT reimplement the volume renderer. Treat it as a pluggable function
behind the documented interface below; stub it and I will wire in the real
call.

## Core architectural decision — read before writing compositor code

Do NOT use Blender's Z-Combine node. A single per-pixel "depth" for a
volumetric pass is not physically meaningful the way it is for two solids —
Z-Combine's hard switch produces artifacts wherever a solid partially
occludes/interpenetrates the volume.

Instead: feed Cycles' Z-depth pass INTO the volume renderer as a per-pixel
integration bound, then composite with plain premultiplied Alpha Over.
Per-frame order of operations:

1. Render Cycles pass first (main scene, flagged volume objects hidden from
   it), with `view_layer.use_pass_z = True`.
2. Extract its Z (depth) pass.
3. Call the external volume renderer, passing camera params + per-pixel depth
   as a far-clip so it stops integrating transmittance at solid surfaces
   instead of rendering "through" them.
4. Composite external RGBA over Cycles beauty via premultiplied Alpha Over.
   No Z-Combine node anywhere in the graph.

Known, explicitly documented v1 limitation: this does not handle a solid
fully enclosed by volume on both near and far sides along the same ray (needs
deep compositing / multiple depth intervals — out of scope now, note it in
the README for future work).

## external_render.py interface contract

Implement this as the hard boundary between the add-on and my renderer.
Use dataclasses (or TypedDicts, your call, but be consistent) matching this
shape — stub the actual call, but the shape must be exactly this so I can
wire in the real renderer without renegotiating the interface:

    @dataclass
    class CameraParams:
        matrix_world: list[list[float]]   # 4x4, camera-to-world
        projection: str                   # "PERSP" | "ORTHO" | "PANO"
        lens_mm: float | None              # perspective only
        sensor_width_mm: float
        sensor_height_mm: float
        sensor_fit: str                    # "AUTO" | "HORIZONTAL" | "VERTICAL"
        shift_x: float
        shift_y: float
        ortho_scale: float | None          # ortho only
        clip_start: float
        clip_end: float
        resolution_x: int
        resolution_y: int
        pixel_aspect_x: float
        pixel_aspect_y: float

    @dataclass
    class DepthBuffer:
        data: "numpy.ndarray"   # shape (resolution_y, resolution_x), float32
        depth_convention: str   # "EUCLIDEAN" | "PLANAR" -- see verification step below
        no_hit_value: float     # sentinel for background/sky pixels; must NOT
                                 # be treated as a real occluder
        row_order: str          # "TOP_LEFT_ORIGIN" | "BOTTOM_LEFT_ORIGIN"
        units: str               # e.g. "meters" -- scene unit scale

    @dataclass
    class VolumeObjectParams:
        matrix_world: list[list[float]]   # 4x4, object-to-world
        bounding_box_local: list[list[float]]  # 8 corners, object-local space
        custom_params: dict   # passthrough from the PropertyGroup:
                               # density scale, absorption/scattering coeffs,
                               # VDB/grid file path, color, etc.

    @dataclass
    class FrameParams:
        frame_number: int
        scene_time_seconds: float

    @dataclass
    class RenderRequest:
        camera: CameraParams
        depth: DepthBuffer
        volume: VolumeObjectParams
        frame: FrameParams

    @dataclass
    class RenderResult:
        rgba: "numpy.ndarray"   # shape (resolution_y, resolution_x, 4), float32
        alpha_mode: str         # "PREMULTIPLIED" -- fixed, must match compositor
        color_space: str        # "LINEAR" -- fixed, must match compositor
                                 # (no view transform applied by the renderer)

    def render_volume(request: RenderRequest) -> RenderResult:
        """Stub — calls out to the real external volume renderer."""
        raise NotImplementedError

No lighting/shadow data is passed in either direction for v1. Do not add
light lists, shadow maps, or irradiance data to this contract — the renderer
is fully self-emissive for now. Note in the README that this is a known
future extension point (e.g., a per-light shadow-map-style depth pass) if we
want Cycles-cast shadows/bounce light on the volume later.

## Required architecture

1. **Marking objects**
   - PropertyGroup flagging an object as externally rendered, storing the
     `custom_params` dict/fields that get passed straight into
     `VolumeObjectParams.custom_params`.

2. **Secondary scene for isolation**
   - Hidden secondary Scene mirroring the main scene's camera/world, linking
     (not duplicating) only flagged objects, so parameter extraction has
     clean access to their transforms without needing them present in the
     Cycles pass.
   - Main scene stays on Cycles; flagged objects hidden from it (visibility
     toggle, not deletion) so its Z pass reflects solids only.

3. **Depth pass extraction — verify before trusting**
   - Before wiring this up for real, build a small test scene (single mesh
     at a known distance from camera) and confirmably determine:
     (a) whether Blender's Z pass is Euclidean ray distance or planar
         (view-axis) distance, and
     (b) the pixel row order convention.
   - Populate `DepthBuffer.depth_convention` and `.row_order` from this
     empirical check, don't assume. Document the check and its result in the
     README.
   - Implement the read-back via a File Output node (EXR, full float) rather
     than assuming in-memory pixel access to render passes is available;
     document whichever approach you use.

4. **Render orchestration**
   - Wrap render triggers (`render_pre` / `render_post` / `render_write` /
     `render_cancel` handlers, or an overridden render operator) so one F12
     press:
     a. Runs the Cycles render of the main scene with the Z pass enabled.
     b. Reads back the Z pass into a `DepthBuffer`.
     c. Builds a `RenderRequest` per flagged volume object and calls
        `render_volume()`.
     d. Writes each `RenderResult.rgba` to a temp EXR (linear float,
        premultiplied alpha, per the contract above).
     e. Programmatically builds/updates a Compositor node graph (match my
        existing node-scripting idiom: `node_tree.nodes.new(...)`,
        `.links.new(...)`, no hand-edited .blend files): Render Layers →
        Image node (reloaded fresh each frame, not cached) → premultiplied
        Alpha Over → Composite / File Output.
     f. Restores object visibility after render, including on cancel/error
        (try/finally or render_cancel handler).
   - Must work for both single-frame and animation renders — depth
     extraction, renderer call, and compositor image reload all need to
     re-run correctly per frame with the right frame number threaded through
     `FrameParams`.

5. **Modularity**
   - `properties.py`, `scene_sync.py`, `external_render.py` (the dataclasses
     above + stubbed `render_volume()`), `compositor_build.py`,
     `operators.py` / `handlers.py`, `__init__.py`.
   - Keep the transport swappable (in-process call now; subprocess or
     network call later) — `render_volume()` should be the only place that'd
     need to change.

## Open questions to resolve explicitly, not silently

1. Multiple flagged volume objects in one frame: composite each
   independently against the same Cycles Z pass, then Alpha-Over them onto
   each other in camera-distance order? Propose a scheme and note it doesn't
   handle volume-through-volume self-occlusion.
2. Whether to pass raw camera intrinsics/extrinsics (as in the dataclass
   above) and let the renderer derive rays itself, vs. precomputing a
   per-pixel ray-origin/ray-direction grid on the Blender side. Pick one,
   document why, keep `CameraParams` as specified regardless (add a
   precomputed ray-grid field alongside it if you go that route, don't
   replace it).

## Out of scope for v1
- Viewport/Eevee preview — final and animation render (F12) only.
- Motion blur matching.
- Deep compositing / multi-interval solid-inside-volume occlusion.
- Any lighting/shadow interchange between Cycles and the volume renderer.

## Deliverable

uv-installable source for an add-on, README documenting: the Z-pass convention
findings from the verification step, all interface assumptions, and a test
scene description to verify (a) volume behind a solid mesh, (b) volume in front
of a solid mesh, (c) a solid partially poking into the volume from one side —
confirming depth-clipped integration handles (a)-(c) correctly, and visibly
demonstrating the known (d) fully-enclosed-solid limitation for future work.
