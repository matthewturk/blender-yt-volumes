# 360° and VR renders

yt_idv modes render every Blender panoramic camera, including spherical stereo
for headsets. This is a recipe for a stereo 360 video. The README's
"Panoramic cameras" section covers how it works.

Use Cycles. EEVEE renders panoramas too, but its panorama depth is
approximate, so the volume is clipped slightly wrong near geometry.

## The camera

Camera Properties > Lens:
- **Type** Panoramic, **Panorama Type** Equirectangular. Fisheyes, mirror
  ball, central cylindrical and cubemap faces work too, but headset players
  expect equirectangular.
- Leave the latitude and longitude ranges at their defaults (the full sphere).

Output Properties: a 2:1 resolution, e.g. 4096 × 2048 per eye for a headset,
or 2048 × 1024 for drafts.

## Stereo

1. Output Properties > **Stereoscopy** on, **Stereo 3D**.
2. Camera Properties > Stereoscopy:
   - **Spherical Stereo** on. Without it each eye is a shifted copy of the
     whole panorama, which only looks right straight ahead.
   - **Mode** Parallel. Off-Axis and Toe-in are for flat screens.
   - **Interocular Distance** is in scene units, read as metres: the default
     0.065 makes one scene unit feel like one metre. A larger distance makes
     the data feel smaller, like a model on a table; a smaller one makes it feel
     huge. Pick it for the scale you want the viewer to feel, not the data's
     physical size.
   - **Pole Merge** fades the eye separation to zero between From (default
     60°) and To (75°) above and below the horizon. Without it, looking
     straight up or down shows swapped or twisted eyes. Leave it on.

Each eye renders the volume separately, so stereo doubles the render time.

## A comfortable camera path

A 360 viewer turns their own head, so the camera shouldn't do it for them.
- **Keep the horizon level.** No roll, and no pitch: the viewer looks up and
  down themselves.
- **Turn only around the vertical.** Start from a level camera (rotation X
  90°, Y 0°) and add a **Locked Track** constraint toward what it should
  face, with **Lock** Y and **To** -Z. The camera then only yaws.
- **Move steadily.** Constant speed, gentle starts and stops, and no shaking.
  Acceleration is what makes people sick, more than speed.
- **Everything around the camera shows**, including what's behind it. A path
  made for a flat camera can pass close to structure that was off-screen.

To reuse a flat camera's path, add a second camera with a **Copy Location**
constraint targeting the first one, and no rotation constraints. Its
**Influence** blends between the first camera's position (1) and the
second camera's own position (0), so placing the second camera at the data's
centre and setting Influence to 0.5 follows the same path, closer in.

## Rendering

In Blender, Render Animation writes whatever the Stereoscopy output is set to.
For a headset, set Output Properties > Views Format to Stereo 3D and **Stereo
Mode** Top-Bottom: left eye on top, as 360 video players expect.

From the command line, `render_volume` sets it with `blender.stereo`:

```toml
[blender]
camera = "Spherical Camera"
resolution = [4096, 2048]   # per eye
stereo = "TOPBOTTOM"        # or SIDEBYSIDE, INDIVIDUAL (a file per eye), OFF
```

Then encode the frames. Many headset players recognize the layout from a
`_360_TB` file name:

```
ffmpeg -framerate 30 -start_number 1 -i frame_%04d.png -c:v libx265 -crf 18 \
       -pix_fmt yuv420p -tag:v hvc1 name_360_TB.mp4
```

Video sites such as YouTube need spherical metadata in the file instead; add it
with Google's [spatial-media](https://github.com/google/spatial-media) tool.
