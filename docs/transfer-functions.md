# Transfer functions

Three of the renderer modes color the volume with a transfer function: the
Geometry Nodes Transfer Function, yt Layers and yt Transfer Function. This page
covers all three, plus the controls they share.

## Geometry Nodes transfer function

**Add > yt Volume** in the 3D viewport makes a flagged Empty-Cube with a
Geometry Nodes transfer function already on it. Its modifier passes its input
geometry through and adds instances of nothing (so nothing draws or renders),
in order from the low to the high end of the transfer function's range (log space),
carrying a `yt_tf_color` color and a `yt_tf_alpha` opacity, plus an
edges-only outline of the Empty's cube or sphere: with a modifier's geometry,
the viewport no longer draws the Empty, and loose edges don't render. Drivers
keep the outline matched to the Empty's display size and type. The starter
group's Color Ramp and opacity curve are also drawn in the Volume Override
panel, and any node setup (math, Scene Time, keyframes) that stores those
attributes works.

The transfer function's x runs linearly in log10 of the value. With **Fit to
Data** (on by default) it spans the data between the **Low %** and **High %**
percentiles by volume (10 and 100); off, Min/Max (log10) set it in the field's
own units. yt Layers uses the same range settings. The ramp's color is weighted by the opacity curve, as in
yt, so what is transparent doesn't glow. **Opacity Scale** multiplies emission
and opacity per box width of path: thin or small structures need it large.
While zoomed, **Opacity Per View** (on by default, also for the yt modes' Ray
Length) keeps that per width of the view, as a yt camera does, so a zoomed-in
look matches the unzoomed one; off, it stays per length of data. After a render the
starter group's points also carry the data's distribution, for viewer node
groups to plot: `yt_tf_volume_fraction` and `yt_tf_mass_fraction` (each
point's share of the volume and the mass) and `yt_tf_log10_value` (log10 of
the field there, in its own units, which the panel also shows as a range). Other objects can read them through Object Info and
Instances to Points, e.g. to draw the transfer function as a plot. The group
uses only built-in nodes, so the .blend opens and evaluates without the
add-on.

## yt Layers and yt transfer functions

**Auto Exposure** (on by default for yt Layers, yt Transfer Function, Geometry
Nodes Transfer Function) scales the transfer function's brightness, not its
opacity, so the brightest 0.1% of pixels sit near white. It costs no extra
render: yt_idv renders into a float framebuffer at the last render's gain, and
since the image is linear in that gain, the image itself is rescaled exactly
and the new gain kept for the next frame. The panel shows the gain. Opacity stays with Ray Length or Opacity
Scale; scaling it too would darken the image past a point, as front layers turn
opaque.

**Show Transfer Function** (offered when yt and matplotlib are installed) draws
the mode's transfer function in the panel with yt's `ColorTransferFunction.show`:
color across the value range, under the opacity curve. It redraws a moment after
the look changes. yt Layers and the Geometry Nodes mode plot over the range Fit
to Data found at the last render, or Min/Max before one. Fade Around Camera
depends on the view and isn't shown.
The title gives the exact log10 range. **Open in Image Editor** shows the
table itself as a float image, one column per bin: the top 16 rows hold each
column's log10 field value and the rows below its RGBA, so clicking a pixel
reads out exact values (display is color only; the readout includes alpha). A
300 dpi copy of the plot, `yt TF plot <object>`, is there to zoom into. Both
update with the plot.

**yt Layers** (`YT_IDV_TRANSFER_FUNCTION`) is `add_layers` without writing a
file: the panel's Layers, Layer Width (`w`), Alpha First/Last (spaced
geometrically, as `alpha=np.geomspace(first, last, N)`) and Colormap build

    tf = yt.ColorTransferFunction((low, high), grey_opacity=True)
    tf.add_layers(N, w=width, alpha=alpha, colormap=colormap)

over log10 of the field in its own units; **Brightness Ramp** then scales the
colors (not opacity) up to that many times brighter at the high end, so small
dense cores glow. **Fade Around Camera** fades out layers the camera has
passed, those sparser than the field it has reached, over **Fade Decades**, so a
dive passes through shells instead of seeing them from inside. Layers ahead
keep their full strength. With a zoom target, the field reached is the highest
along the line in from the domain edge, so the fade never backs up. **Fit to Data** takes `(low, high)`
from percentiles of the data by volume (default 10% to 100%; the panel shows
the range it found); off, it uses Min and Max (log10). It shares Ray Length
with the mode below, and is converted the same way.

**Opaque End** adds a solid end after the layers: from **Opaque Above** (log10)
up the field is opaque, in the colormap's color there, its opacity rising
geometrically from Alpha Last over **Opaque Edge** decades below, so dense gas
turns solid gradually. Fade Around Camera leaves it alone, and Auto Exposure's
gain fades out along the rise and is measured only where the end hasn't made
the image half opaque, so the solid end keeps its own color and the layers
around it stay exposed as without it. It needs a
yt_idv whose transfer-function shader integrates each step exactly
(yt-project/yt_idv#312, newer than 0.6.1); otherwise its first sample
overshoots to white.
The plot shows opacity only up to 1, so the rise looks like a step there.

`render.mode = "YT_IDV_YT_TRANSFER_FUNCTION"` renders with a transfer function
you build in yt, a `ColorTransferFunction` with grey opacity over log10 of
the field. It is converted with yt_idv's `tf_conversion` (yt-project/yt_idv#288),
so it renders as yt would. Name the function in `[render.yt_idv]`:

    yt_tf = "my_tf.py:make"      # or "module:function"
    yt_tf_ray_length = 0.03      # divides emission and opacity; default 0.02
    samples_per_cell = 8         # set to 16 on choosing a yt mode, 4 for the others

`make()` with no arguments is converted once. `make(context)` is called every
frame with `frame`, `camera` (the camera's unitary float64 position) and `ds`,
so a transfer function can follow the camera. `examples/shells_tf.py`
hides isodensity layers sparser than the gas around the camera, which turns
the DeeplyNestedZoom dive (`examples/deep_zoom_shots_tf.toml`) into a passage
through nested shells, one per decade of zoom. In a zoom the near and far
planes are bounded to 1e-3 and 1e3 times the camera's distance to the target:
yt_idv rebuilds rays from the inverse projection in float32, and a domain-wide
far plane there bands transfer-function renders.
