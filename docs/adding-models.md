# Adding your own models

Any mesh in the scene, modeled in Blender or imported (STL, OBJ, glTF, ...), is
rendered by the scene's engine in pass 1, and the volume is clipped at its depth.

**What works**
- A model in front of the volume hides the volume behind it.
- Volume in front of a model is drawn over it, so the model shows through the haze.
- A model poking into the volume, or fully inside it, gets a clean boundary
  along its near surface; the volume behind it is cut off.

**Limits**
- Only occlusion is shared. The volume doesn't light, shadow or reflect in models,
  and models don't shadow the volume.
- Semi-transparent models: in EEVEE use the Dithered render method (Blended writes
  no depth, so the volume draws over it); in Cycles, View Layer > Passes > Data >
  Alpha Threshold decides whether a model hides the volume. Either way the volume
  behind it is all or nothing per pixel, not partly visible.

**Looking right**
- Place and scale models in data coordinates with the volume's Units Object (see
  "Scene units per data unit" in the README), not by eye.
- Imported models often lack smooth normals and materials: Shade Smooth (or Auto
  Smooth) and assign one.
- With no lights in the scene, give models emission or their own lights, or they
  render black.
- To block the volume without being seen (a mask), use a holdout.
- Deep zooms: meshes are float32. Far from the origin at large zoom they jitter or
  lose detail while the volume doesn't; keep detailed models near the zoom target.

**Performance**
- Decimate models to what the camera can resolve; millions of triangles cost
  memory and time in both engines.
- Static models are cheap in Cycles animations (persistent data keeps them
  between frames); animated or deforming ones re-sync every frame.
- Smooth Surface Edges keeps 2 objects per pixel, so where three or more overlap
  the edges are a little less exact.
- Large models and a large dataset share the GPU: see [render-speed.md](render-speed.md).
