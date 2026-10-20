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

## Exposure

Turn on **Auto Exposure** (the default): each frame's brightness is scaled so its
brightest 0.1% of pixels sit near white, exactly and with no extra render. Ray Length then only sets opacity, how
see-through the shells are. With **Opacity Per View** (the default) it applies
per width of the view, so one value gives the same look at every zoom; off, it
is per length of data, and deep zooms need far smaller values.

Without it, Ray Length is also the exposure, and has to be calibrated by zoom
level: render every 25th frame at a ladder of Ray Lengths, keep the ones that
clip at most ~0.5% of pixels, pick the one with the most core contrast (the
99.9th percentile of brightness over the median), and keyframe a smooth curve
through the picks. Brightness isn't monotonic in Ray Length: past a point, a
shorter one makes the front shells opaque and the image darker.

## Checklist and costs

- Use a yt_idv with its compositing fixes (blocks drawn nearest first and
  integrated front to back; premultiplied color passed through as is). Without
  them, thick zoomed views composite as seen from behind, and Auto Exposure
  can't brighten thin gas.
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
exposure = "auto"
ray_length = 4e-4
fps = 24
```

`zoom_movie` would render with those settings, record the exposure each frame
chose (so it can be inspected or pinned), and encode the MP4.
