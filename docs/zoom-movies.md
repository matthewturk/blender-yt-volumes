# Making a zoom movie with yt Layers

A recipe for a movie that flies from the whole box into a dense core, worked out
on DD0087 (`dd0087_scene4.zip`, 250 frames from zoom level 0 to 4.2 toward
`"max"`). None of this is automated yet; the last section sketches what a
scripted version would take as input.

## What makes it read as flying in

- **Zoom with the volume's own zoom.** Set **Zoom Target** (`"max"` or a
  unitary point) and animate **Zoom Level** linearly: a steady fly-in. Magnifying
  about the target with a fixed camera is the same, geometrically, as flying the
  camera straight at it.
- **Keep the transfer function fixed.** Use one set of layers for the whole
  movie (Fit to Data off, Min/Max in log10 spanning everything the movie passes
  through, e.g. -26.5 to -15 with 24 layers). Shells then keep their density and
  color, grow as you approach, and pass behind you. Moving the bounds with the
  zoom makes gas fade out and get replaced instead.
- **Fade the shells you've passed.** Turn on **Fade Around Camera**. With a zoom
  target, the camera's density is the highest met on the way in (smoothed), so
  the fade only moves forward; shells ahead keep their full strength, and a shell
  fades over **Fade Decades** (1 works) once the camera is through it.
- **Let the core glow.** A low **Alpha First** (0.002) keeps the sparse shells
  see-through, and a large **Brightness Ramp** (100-300 over ~10 decades) makes
  the dense, small inner layers bright.
- **Leave Limit Zoom Depth on** (2x). Otherwise rays run through the rest of
  the box behind the target and the sparse far side hazes the view.

## Exposure: Ray Length has to follow the zoom

Opacity is per data length, so Ray Length is the exposure. As the view shrinks,
the right Ray Length falls by several decades, and not at a fixed rate: it
depends on how much gas surrounds the camera. It isn't monotonic in brightness
either: past a point, a shorter Ray Length makes the front shells opaque and the
image darker.

Calibrate it rather than guess:

1. Render every 25th frame at low resolution (640x360 is plenty) at a ladder of
   Ray Lengths, half a decade apart, around a rough guess
   (`2e-3 * 10**(-0.56 * level)` for DD0087).
2. For each probe frame, keep the settings that clip at most ~0.5% of pixels,
   and pick the one with the most core contrast: the 99.9th percentile of
   brightness over the median. Picking on mean brightness lets the fog set the
   exposure and dims the core.
3. Fit a smooth curve through the picks (log10 of the multiplier against zoom
   level) and keyframe Ray Length from it every few frames. Per-frame picks jump
   by up to 300x between neighbors and would flicker.
4. Render every 25th frame with the curve and check clipping before the full
   render; nudge the whole curve brighter or darker if there's headroom.

Each probe frame takes ~1 s once the scene is loaded, so a full ladder is a few
minutes. Recalibrate whenever the layers, alpha, ramp or fade change.

## Checklist and costs

- Render with yt_idv compositing front to back (the draw-order fix): without it,
  thick zoomed views composite as seen from behind.
- `render_volume` sets the zoom target and level from `[zoom]` every frame, so
  give the level as a curve there (`level = [[1, 0.0], [250, 4.2]]`), not only as
  keyframes.
- DD0087 at 1920x1080: ~105 s to load the 1.8 GB scene, then ~3.2 s a frame;
  250 frames take ~15 minutes and ~600 MB of PNGs.

## What a scripted version would take

The steps above are fixed; only the inputs vary. A `[movie]` section and a
`zoom_movie` command could run them:

```toml
[zoom]
target = "max"
level = [[1, 0.0], [250, 4.2]]

[movie]
layers = { log_range = [-26.5, -15.0], count = 24, alpha_first = 0.002, brightness_ramp = 300 }
fade = { decades = 1.0 }
exposure = { probe_every = 25, clip_max = 0.5, guess = "2e-3 * 10**(-0.56 * level)" }
fps = 24
```

`zoom_movie` would probe, calibrate and smooth the Ray Length curve, write it to
a report (so it can be inspected or pinned), render, and encode the MP4.
