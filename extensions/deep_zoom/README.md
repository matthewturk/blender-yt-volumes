# Deep Zoom

A Blender extension for authoring camera paths through deeply nested AMR data,
the kind where the interesting structure is 10⁻¹² of the domain across.

Blender holds positions in float32, so a camera flying toward a tiny feature
stops being able to say where it is around 10⁻⁶ of the box. Deep Zoom never
asks it to: the scene is always in **shot space**, where

    world = (unitary − anchor) × 10^zoom × world_scale

The anchor is a float64 point kept as text, and the zoom is an ordinary
animatable number. The anchor is always the world origin, where float32 keeps
its relative precision, so the camera moves at ordinary sizes whatever the
depth, and you dive by animating the zoom.

## Install

Zip the folder and use *Edit > Preferences > Get Extensions > Install from
Disk*:

    cd extensions/deep_zoom && zip -r ../deep_zoom-0.1.0.zip . -x '*__pycache__*'

It needs nothing beyond what Blender ships (numpy).

## Workflow

1. **Export the grid structure** with any Python that has yt:

       python grid_structure.py /path/to/dataset grids.npz

   This writes the grids' float64 edges and levels. Point the panel's *Grid
   Structure* at it.
2. **Turn on Shot Space** in the *Deep Zoom* sidebar tab (N panel) and add a
   first shot. Its anchor defaults to the domain center; type in any unitary
   `x, y, z` (or `max`, for the renderer to find the field's peak).
3. **Dive.** The ▼ button next to *Zoom* keyframes a steady exponential dive:
   N decades over M frames, linear in the zoom. Animate the camera as usual,
   a few units from the origin.
4. **Steer by the grid map.** The *DeepZoomGridMap* object shows the grid
   boxes that are near the current scale (between *Smallest* and *Largest*
   world units, within *Reach* of the anchor, the *Max Boxes* nearest it),
   colored by level. It is
   rebuilt from float64 every frame, so at zoom 12 you still see the level-40
   grids at a size you can work with. It renders in Cycles too.
5. **Re-anchor** where you want the dive to go: put the 3D cursor on it and
   press **+** (*New Shot at Cursor*), or use *Re-anchor at Focus* to anchor at
   whatever the camera tracks. A re-anchor is a cut. Earlier frames stay
   exactly as they were (the curves are split shape-preserving), and the view
   doesn't jump. From then on the camera's motion is unchanged in shot space,
   and the dive converges on the new point.
6. **Check the path.** *Check Path* scans every frame and drops timeline
   markers where you should
   - **re-anchor**: float32 error in the coordinates in play exceeds ¼ pixel
     at the focus distance, i.e. the camera and what it looks at have drifted
     far from the anchor relative to the view;
   - **rescale**: the focus is closer than 0.01 or farther than 100 world
     units. *Rescale* changes units from the current frame on: the zoom rises
     by N decades and the camera and its target move 10ᴺ farther out, so the
     image doesn't change.

   The panel also shows the current frame's status.
7. **Export** writes every frame's camera pose (position, forward, up) in
   unitary float64, plus the shots and lens, as JSON. That's the exact path,
   independent of Blender's precision, for any renderer to consume.

## Rendering

yt-volumes renders flagged volumes in shot space whenever this
extension is enabled: it reads the scene's `deep_zoom` settings and builds the
volume's registration in float64. Headless, list the extension in the config:

    [blender]
    extensions = ["path/to/extensions/deep_zoom"]

## Files

- `shot_space.py`: the float64 mapping and grid-box selection; pure numpy.
- `grid_structure.py`: the `.npz` exporter (needs yt) and loader (numpy).
- `authoring.py`: shots, the grid map, dive, re-anchor and rescale cuts, the
  path check, export, and the UI.
