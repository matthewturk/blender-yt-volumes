---
aliases:  ["blender - 2026-10-20 - yt cycles demo script v2"]
tags:
 - projects/blender/note
 - notes  
created: 2026-10-05  
project: blender
modified: 2026-10-20
---

# Blender - 2026-10-20 - yt Cycles Demo Script v2

A read-along script for a screen-capture walkthrough. The example throughout is `biggest_atlas.zip`, PopII_prime's saved yt_idv scene (`biggest_scene.zip` re-saved in the texture-atlas format; the old file no longer loads): 94,260 blocks, 26 refinement levels, cells down to 1.5e-11 of the box. Everything here was run on 2026-10-04 and works (interactively through the F12 path, and in batch), except the viewport grid overlay (Scene 3), added 2026-10-06 and not yet tried in the Blender GUI. Updated 2026-10-20 for the current add-on: the menu names, the new defaults (**Auto Exposure** and **Opacity Per View**, both on), and the newer pieces this script now shows: the live viewport preview, the yt Layers mode with Fade Around Camera, and the transfer-function plot. Those new parts haven't been rehearsed on this scene; run through them once. Timings are from an RTX 2060 at 960x540 unless noted, and predate the atlas, which made renders faster; re-time them.

Each scene has:
- **DO**: what to click or type
- **SAY**: suggested narration (paraphrase freely)
- **NOTE**: things to know or say honestly

The story in one line: *author the camera move interactively in Blender at human scales, let the engine handle the 10 orders of magnitude, then hand the same .blend to a batch renderer; or go the other way, and open a batch-planned path in Blender to edit it.*

Runtime: about 25-30 minutes, or 12 with the short version at the end.

`$REPO` below is the yt-volumes checkout; `$SCENE` is `/data/PopII_prime/biggest_atlas.zip`.

---

## Before You Hit Record

1. **Export the grid structure** (3 seconds, numpy only), for the scene builders below; Blender itself reads the same blocks straight from the `.zip`. Don't skip it: `popii_blocks.npz` isn't there as of 2026-10-20, and step 2 and Scene 11 need it.  
       `python $REPO/extensions/deep_zoom/grid_structure.py --scene $SCENE $REPO/examples/popii_blocks.npz`

2. **Build the fallback .blend** (a finished version of what you'll author  
   live, in case anything goes sideways on camera):  
       `python $REPO/examples/make_deep_zoom_shots_blend.py $REPO/examples/popii_dive.blend \`  
           `$REPO/examples/popii_blocks.npz --dataset $SCENE --first-anchor bbox \`  
           `--zooms 1.3 8 9 --cut 150`

   and a no-grid-map copy for the batch section (`... popii_dive_nogrid.blend  
   ... --no-grid-map`).
3. **Warm the cache.** The first F12 on `biggest_atlas.zip` asks *Load data to render?* (confirm with **Load and Render**), then loads all 94,260 blocks: about **2 minutes** before the atlas, probably less now. After that each F12 takes 8-15 s. Do one throwaway F12 off camera and keep that Blender open for the interactive sections; it also loads the grids for the viewport overlay. The loaded scene is cached per volume object, so warm up the Empty you'll use on camera: a new Empty pays the load again, a Shift+D duplicate doesn't. In Scene 2, walk through its settings rather than creating it live.
4. **Render the batch movie ahead of time** (Scene 10 shows it starting live; you won't want to wait on camera):

       python -m yt_volumes.render_volume $REPO/examples/popii_dive.toml \
           --blend $REPO/examples/popii_dive_nogrid.blend \
           --output /data3/scene-extract/popii_dive/frame_

   About 2 minutes to load, then ~1.5 s per frame at 1280x720 (re-time: Auto Exposure can re-render a frame up to four times while it settles, mostly in the first frames). The frames go to `/data3` because `/home` is nearly full. Then encode them to the movie Scene 10 cuts to; `popii_dive_shells.mp4` in the cutaways predates the new defaults and looks different.
5. **Cutaways** in `~/misc-transfer/demo_assets/`: `popii_dive_shells.mp4` (the finished PopII dive), `deep_zoom_shells_tf.mp4` (DeeplyNestedZoom, 12 decades), `deep_zoom_shots_side_by_side.mp4` (with/without the grid map), `earth_plumes_spin.mp4` (spherical data), and `popii_interactive_frames.png` (what interactive F12 frames look like).
6. Blender:
   - Preferences > System > **GPU Backend: Vulkan**, then restart. Under OpenGL, yt_idv can't get its GPU context in the GUI. Check it stuck: a dotfile setting `gpu_backend: OPENGL` can undo it on restart.
   - Render Engine **Cycles**. EEVEE also works under Vulkan, but this script was rehearsed with Cycles.
   - Render > Sampling: low max samples (16) and the **OptiX** denoiser. A fresh scene's 4096 samples with CPU denoising take ~50 s per pass, and every F12 is two passes.
   - View Transform **Standard**; the add-on and the **Deep Zoom** extension enabled.
   - Render with **F12** or *Render > Render (with Volume Override)*. Blender's own *Render Animation* leaves the volumes out; it warns, but don't use it on camera.

Disk: `/home` was 98% full on 2026-10-20 (21 GB free); `/data3` has ~470 GB. A 240-frame 1280x720 PNG render is ~0.5 GB.

---

## Scene 1: Cold Open (30 S)

**DO:** Play the batch movie you rendered in *Before You Hit Record* (or the older `popii_dive_shells.mp4`).

**SAY:** "This is a Population II star-forming region, rendered in Blender. The camera starts with the whole region in view and ends looking at cells a ten-billionth of the box across. That's about eight orders of magnitude, and Blender keeps positions in 32-bit floats, which give out after seven. The volume is yt_idv, a GPU volume renderer from the yt project, composited into Cycles with real depth. I'll show how that works, how you author a move like this interactively, and how you hand it to a batch renderer."

---

## Scene 2: A Saved Scene, One F12 (2 Min)

**DO:**
1. Add > Empty > Cube. Object Properties > **Volume Override**: check **Render with External Volume Renderer**, set **Dataset Path** to `biggest_atlas.zip`.
2. In the same panel: **Renderer** = *Maximum Intensity Projection*.
3. Point the camera at the cube and press **F12**.
4. Click **Show in Viewport**. The volume now draws in the 3D viewport, behind and in front of the scene's objects. Orbit: it re-renders a moment after you stop moving.

**SAY:** "The Empty cube is a stand-in: Cycles never sees it, but its box is where the data goes. The dataset path can be anything yt loads, or, like here, a scene yt_idv saved to a zip, which is just the GPU-ready blocks, so Blender doesn't need the original simulation at all. Two passes happen on F12: Cycles renders the scene and its depth, then yt_idv renders the volume from the same camera, clipped to that depth, and the two are composited. The renderer and all its settings live on the volume and are saved in the .blend, so the batch renderer later produces the same look. And the same renderer drives the viewport: Show in Viewport renders the volume from wherever I'm looking, occluded by the scene, so I can frame shots without pressing F12."

**NOTE:** If this F12 is cold it asks to load, then takes minutes; that's why you warmed it up. Under the OpenGL GPU backend with EEVEE you get an error telling you to switch to Vulkan or Cycles. **Viewport Resolution** (beside Show in Viewport) trades sharpness for speed; 0.5 is the default.

---

## Scene 3: Where's the Refinement? (2 Min)

**DO:**
1. In the Volume Override panel, check **Show Grids in Viewport**. Boxes appear in the viewport, blue for coarse levels and teal for fine ones, and the panel shows the level range.
2. Orbit around. Raise **Min Level** to about 15: the coarse boxes go, the deep ones stay.
3. Check **Consolidate**, **Boxes** 4: the deep levels collapse into a few tight boxes around where the refinement actually is.
4. Press F12: the boxes aren't in the render.

**SAY:** "The volume only exists at render time, but its block structure is cheap to draw. This is the viewport, live, showing where the simulation put its resolution. Up to five thousand boxes, shared across the selected levels so the deep ones aren't crowded out. With Consolidate on, it draws a handful of boxes around the deep levels instead of thousands, which tells you where to point the camera. It only draws in the viewport, so it doesn't change the render."

**NOTE:** This scene is untested in the GUI; rehearse it. Grids show up only after the volume has rendered once in the session (the warm-up covers it). For a saved scene the boxes are yt_idv's blocks, not the original AMR grids, with levels taken from their cell sizes. Boxes are clipped to the Empty, and a box covering all of it isn't drawn, since it would just be the Empty's outline.

---

## Scene 4: Render Modes in Three Minutes

**DO**, re-pressing F12 after each (or just watching the viewport preview):
1. **Rescale Colormap Every Render**: check it (it's under the renderer dropdown). Mention why: the dive spans 20 decades of density, and a colormap range fitted at one depth clips another.
2. Renderer = *Projective Integration*.
3. Renderer = *yt Layers*. **Samples per Cell** jumps to 16 by itself. Uncheck **Fit to Data** and set **Min (log10)** -27, **Max (log10)** -10.6 (the PopII density range), **Layers** 12, **Colormap** arbre. Leave **Auto Exposure** and **Opacity Per View** on.
4. Check **Show Transfer Function**: yt's own plot of these layers appears in the panel. Change **Layers** to 6 and back: the plot follows.
5. Renderer = *yt Transfer Function*. Set **yt Transfer Function** to `$REPO/examples/shells_tf.py:popii` and **yt TF Ray Length** 0.03. Open `shells_tf.py` in a text editor.

**SAY:** (on yt Layers) "This is yt's `add_layers`, isodensity shells, built from the panel, no code. Auto Exposure brightens each frame until its brightest pixels sit just under white, smoothly from frame to frame, so one setting works from the whole region down to a single cell; the opacity setting only decides how see-through the shells are. And this plot is yt's own `show()` of the transfer function, redrawn as I change it."

**SAY:** (on the file) "When the panel isn't enough, write the transfer function in yt: this is a plain ColorTransferFunction, Gaussian layers in log density, colored with yt's arbre colormap. yt_idv converts it so it renders the way yt would. The function gets a context each frame, including the density at the camera, so it can hide the layers you're already inside; yt Layers does the same with one checkbox, **Fade Around Camera**, which I'll use for the dive."

**NOTE:** The yt Layers values are a starting point; tune **Alpha First** (lower sees deeper through the outer shells) and **Brightness Ramp** (makes the dense inner shells glow) in rehearsal. *Open in Image Editor*, beside Show Transfer Function, shows the table's exact values if you need them. With Opacity Per View on, yt TF Ray Length means opacity per view width rather than per data length, so 0.03 may want adjusting for the file-based mode too.

---

## Scene 5: Depth is Real (1 Min, Optional)

**DO:** Add a UV sphere (normal material, not flagged) so it pokes into the volume. F12.

**SAY:** "The sphere isn't pasted on top: gas in front of it still renders, gas behind it is cut off, because yt_idv stops each ray at the depth Cycles wrote."

**NOTE:** A single depth per pixel clips at one point, so a solid fully inside the gas with gas on both sides of it along a ray isn't carved out in the middle.

---

## Scene 6: Why Deep Zooms Are Hard (1 Min)

**SAY:** "Now the move. The obvious way to fly into a cell this small is to move the camera toward it. Blender holds positions in 32-bit floats, about seven significant digits, so somewhere around a millionth of the box the camera can't say where it is any more. It jitters, then it's simply lost. I measured it: a camera one Blender unit from the origin is 1.5% off at a millionth and gone by a hundred-millionth."

**DO (optional cutaway):** `deep_zoom_shots_side_by_side.mp4`: 12 decades of DeeplyNestedZoom, with and without the grid map.

---

## Scene 7: Authoring the Dive Interactively (5 Min)

This is the heart of it. Everything is in the 3D viewport's sidebar (N), **Deep Zoom** tab. Uncheck the volume's **Show Grids in Viewport** first: Deep Zoom draws its own grid map in shot space, and the two overlap.

**DO:**
1. Check **Shot Space**. Leave **Grid Structure** empty: Deep Zoom reads the blocks from the volume's `.zip`, the same boxes `popii_blocks.npz` holds.
2. Press **+** (*New Shot at Cursor*); the first shot starts at the domain center. Set its **Anchor** to the region's center,  
   `0.5396208, 0.53484833, 0.51514244`, then set **Zoom** to 1.3.
3. The **grid map** appears: the AMR blocks near the current scale, colored by level. Drag **Zoom** up slowly and watch new levels of boxes grow out of the center while the outer ones leave. Point at the readout: "1 unit = ... of the domain".
4. Add a camera a few units out, Track To an Empty at the origin. At frame 1 press the **Dive** button (the down-arrow next to Zoom), then in *Adjust Last Operation* (bottom-left, or F9) set **Decades** 6.7 and **Frames** 149: zoom 1.3 to 8 by frame 150. Key a slow orbit on the camera.
5. Switch the volume to *yt Layers* with **Fade Around Camera** on (**Fade Decades** 1). Scrub the timeline: the viewport preview follows the zoom, re-rendering when you stop. F12 at frames 1, 75, 150.

**SAY:** "In shot space the anchor is always the origin, and Blender's world is the data scaled by ten to the zoom. The camera stays a few units out, where float32 is perfectly happy; the zoom does the diving, and it's in 64-bit. The grid map is the trick that makes this usable: the grid boxes are cheap, so at any depth you can see where the refinement goes and steer by it, and the viewport preview shows the volume itself whenever I stop. Fade Around Camera fades out each shell once the camera has passed through it, so the dive goes through the shells instead of staring at the inside of one."

**DO, continued (the re-anchor):**  
6. Show why it matters: temporarily type 10 into **Zoom**. The finest boxes on the grid map sit well off center: the region's center isn't the deepest point. Put Zoom back (or step back to frame 150).  
7. At frame 150 (zoom 8) press **New Shot at Finest Grid** (the snap icon under + / − / ▶ beside the shot list). A shot "Shot 2" appears starting at 150, anchored on the finest box in 64-bit. Press **Dive** and set **Decades** 1, **Frames** 90: zoom 9 at frame 240.  
8. Press **Check Path**. No markers. Show the status box (focus distance in units, jitter in pixels).

**SAY:** "The region's center and the deepest cell are 4e-10 of the box apart. At zoom 8 that's a twelfth of a unit, invisible; at zoom 10 it's seven units, enough to miss the core entirely. So partway down I start a new shot anchored on the finest grid. A shot change is a cut: frames before it are untouched, the view doesn't jump, and the rest of the dive converges on the new point. Check Path scans every frame and drops a timeline marker wherever 32-bit error would exceed a quarter pixel, or the camera is at an awkward scale, and tells you to re-anchor or rescale there."

**NOTE:** Re-anchor and rescale are both exact for earlier frames (the curves are split shape-preserving). If live authoring gets fiddly, open the fallback `popii_dive.blend`: it's exactly this, built by the same operations. **Anchor at Densest Point** (in the Deep Zoom panel, once the volume has loaded) sets a shot's anchor to the densest cell in 64-bit without typing coordinates; for the region's center in step 2, typing is still right. The viewport preview and the grid map both draw in shot space, so they line up.

---

## Scene 8: Export the Path (30 S)

**DO:** **Export Camera Path** > `popii_path.json`. Open it.

**SAY:** "Every frame's camera, in the dataset's own coordinates, in 64-bit: position, direction, distance. That's the actual path, independent of Blender's precision, so anything can render it."

---

## Scene 9: Interactive Frames

**DO:** F12 at frames 1, 150, 240 in *yt Layers* mode with Fade Around Camera (or cut to `popii_interactive_frames.png`, top row MIP, bottom row shells, from the older version).

**SAY:** "Eight decades of zoom, and every frame is exposed the same way without touching a setting: that's Auto Exposure, and Opacity Per View keeps the shells equally see-through at every depth."

**NOTE:** Turn Deep Zoom's grid map **off** for beauty frames: its boxes are Cycles geometry, so inside the volume they cut the rays behind them and show up as dark lines in MIP. It's great for steering, not for the final frame. The volume's own grid overlay (Scene 3) is viewport-only and can stay on.

---

## Scene 10: Batch: the Same .blend, Headless (2 Min)

**DO:** Show `examples/popii_dive.toml`, then run it:

    python -m yt_volumes.render_volume examples/popii_dive.toml \
        --output /data3/scene-extract/popii_dive_live/frame_

**SAY:** "The batch renderer takes one TOML file: the data, which .blend, which object is the volume, frames, resolution, the render mode and its settings. It registers the Deep Zoom extension, so the shots and zoom you authored come along untouched. After a two-minute load it runs about a second and a half per frame at 720p, and the result is the movie from the start."

**DO:** Let it start, then cut to the finished `popii_dive_shells.mp4`.

**SAY (other batch tricks, 30 s):** "The same config format does time series, one dataset per frame range (`render_series`); takes any Python function as a loader, so synthetic or custom data works; and renders time-varying data by rebuilding the volume each frame without leaking GPU memory. It also handles spherical data: a sphere stands in for the planet." **Cutaway:** `earth_plumes_spin.mp4`.

**SAY (newer, 20 s, mention only):** "Stereo renders work too, each eye with its own volume render; Deep Zoom can carry points of interest along the dive, exact at any depth; and each volume publishes how many scene units a kiloparsec is right now, for drivers and Geometry Nodes."

---

## Scene 11: One Sets up the Other (3 Min)

**Interactive to batch** is what you just saw: author in Blender, save, point a config at the .blend.

**Batch to interactive:**

**DO:**

    python examples/make_flythrough_blend.py $SCENE out/popii_fly --deep-zoom \
        --grids examples/popii_blocks.npz

Open `out/popii_fly.blend` in Blender.

**SAY:** "This goes the other way. `flythrough_densest.py` is a yt_idv script that finds the densest cell and plans a dolly-and-orbit toward it, with a colormap plan to match. This tool runs that planner once, offline, and writes a Deep Zoom scene: one shot anchored on the densest cell, the approach in the zoom curve, the camera keyed in ordinary units. Open it and it's an ordinary editable scene: here's the zoom curve in the Graph Editor, here's the grid map. I can retime it, change the orbit, add shots, Check Path, F12. It renders the same frames the original script did, to within 3/255, and the TOML it wrote next to it batch-renders it."

**DO:** Graph Editor on the zoom curve; nudge a key; F12.

---

## Scene 12: Wrap-up (1 Min)

**SAY:**
- "Interactive and batch use the same pieces: the add-on renders, the Deep Zoom extension authors, and a TOML file turns a .blend into a movie."
- "The precision trick is boring on purpose: never ask float32 to hold a big number. The anchor is 64-bit text, the zoom is a small number, and the camera stays a few units from the origin."
- "It runs on yt_idv's main branch: the scene transforms, the texture atlas and the transfer function conversion all went into yt_idv for this. The atlas packs the blocks into shared textures instead of one per block; on a test scene it cut peak GPU memory from 4.1 to 2.8 GB and the first frame from 122 to 81 seconds."

---

## If You're Short on Time (~11 Min)

Scene 1 (cold open) > Scene 2 (one F12, then Show in Viewport) > Scene 3 steps 1 and 3 (grids, consolidated) > Scene 4 steps 3-4 only (yt Layers and its plot) > Scene 7 steps 1-5 and the re-anchor narration (author the dive) > Scene 10 (batch, cut to the movie) > Scene 11 in narration only, over the Graph Editor > Scene 12.

Cut first: Scene 5 (depth), Scene 8 (export), Scene 4 step 5 (the transfer-function file), the spherical cutaway and the "newer" mentions in Scene 10.
