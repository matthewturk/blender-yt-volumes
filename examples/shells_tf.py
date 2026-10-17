"""yt ColorTransferFunctions of isodensity shells, for deep dives.

Gaussian layers evenly spaced in log10 density, colored with yt's arbre.
Layers sparser than the gas at the camera are hidden (the camera is inside
those shells, and they would fog every ray), and opacity rises with how much
denser a layer is: the shell just ahead is a faint veil the inner ones show
through, and it dissolves as the dive reaches it. Each function takes the
renderer's per-frame context; context["value"] is the density at the camera.

Editing this file
-----------------
Point Preferences > yt Transfer Function at `/path/to/shells_tf.py:popii` (or
`popii_static`). The file is re-read whenever yt_idv actually renders, but
an F12 with nothing else changed reuses the last image. After saving an edit,
click Reset Change Tracking (Volume Override panel) or nudge the camera, then
F12. A no-argument function is also cached until the TF setting or ray length
changes; the functions here take `context`, so they aren't.

What each knob does, and what to expect:

- `layers` (the `np.linspace(lo, hi, n)` in `popii`): one shell per entry, in
  log10 density. More entries = more, closer-spaced shells. Narrow the range
  to spend all the shells on the structure you're looking at, e.g.
  `np.linspace(-24, -18, 8)` for the filaments around the halo.
- `bounds` (the `(-28.0, -10.0)` pair): the TF's whole range. Keep it a bit
  wider than `layers`; densities outside it are invisible.
- `WIDTH`: shell thickness, in (log10 density)**2. Bigger = fatter, softer,
  more opaque shells that merge into fog; smaller = crisp thin skins that can
  break up into speckle. Try 0.001-0.01. If shells look dotted or banded,
  raise Samples per Cell (16 is the yt modes' default) before raising WIDTH.
- `alpha = np.geomspace(0.3, 1.0, ...)`: opacity from the sparsest to the
  densest shell. Lower the first number to see deeper into the volume
  through the outer shells; set both ends to 1.0 for solid shells.
- `FADE` (only in `popii`): decades above the camera's density over which a
  shell fades in. Shells sparser than the camera's gas are always hidden.
  Large FADE = the dive looks nearly empty until you're well out of a shell;
  small FADE = shells pop in abruptly. With the camera outside the volume
  there's no density at the camera, so nothing fades.
- `colormap="arbre"`: any yt/matplotlib colormap name; colors span
  `col_bounds`, so the sparsest shell gets the low end of the map.

What to expect: from outside the box, nested translucent colored skins around
the dense clumps, reddish outer shells, yellow-white inner ones (arbre). With
`popii` and the camera inside the box, the shells around the camera vanish on
purpose; at zoom level 0 the denser shells left are only a few pixels across.
Use `popii_static` for free flying inside the box, or Deep Zoom to magnify.
TF Ray Length divides opacity: yt's TF is per unit of a yt camera's ray
length (in unitary units). The dive uses 0.03; the default is 0.02. 1.0 makes
every shell ~33x fainter than the dive's, which can look like nothing at all.
Halve it to double opacity.
"""

import numpy as np

WIDTH = 0.003  # yt's add_gaussian width, in (log10 density)**2
FADE = 4.0  # decades above the camera's density over which a layer fades in


def shells(context, layers, bounds):
    import yt

    tf = yt.ColorTransferFunction(bounds, nbins=8192, grey_opacity=True)
    alpha = np.geomspace(0.3, 1.0, layers.size)
    value = context.get("value")
    if value is not None and value > 0:
        alpha = alpha * np.clip((layers - np.log10(value)) / FADE, 0.0, 1.0)
    for layer, a in zip(layers, alpha):
        if a > 0:
            tf.sample_colormap(layer, WIDTH, alpha=a, colormap="arbre", col_bounds=(layers[0], layers[-1]))
    return tf


def deeply_nested_zoom(context):
    """Density falls as r**-2 there, so layers two decades apart are one
    shell per decade of radius, and of zoom."""
    return shells(context, np.linspace(-26.0, -4.0, 12), (-27.0, -3.0))


def popii(context):
    """PopII_prime's biggest_scene: density 1e-31 to 5e-11 g/cm**3."""
    return shells(context, np.linspace(-27.0, -10.6, 12), (-28.0, -10.0))


def popii_static(context):
    """`popii` without the camera-density fade: every shell always drawn."""
    return shells({}, np.linspace(-27.0, -10.6, 12), (-28.0, -10.0))
