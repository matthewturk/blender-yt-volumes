"""A yt ColorTransferFunction for DeeplyNestedZoom: isodensity shells.

The density there falls as r**-2 around the peak, so layers two decades apart
are one shell per decade of radius, and a steady dive passes a new one every
decade of zoom. Layers sparser than the gas around the camera are hidden (the
camera is inside those shells, and they would fog every ray), and opacity
rises with how much denser a layer is: the shell just ahead is a faint veil
the inner ones show through, and it dissolves as the dive reaches it.
"""

import numpy as np

LAYERS = np.linspace(-26.0, -4.0, 12)  # log10 density, g/cm**3
WIDTH = 0.003  # yt's add_gaussian width, in (log10 density)**2
FADE = 4.0  # decades above the camera's density over which a layer fades in


def make(context):
    import yt

    tf = yt.ColorTransferFunction((-27.0, -3.0), nbins=8192, grey_opacity=True)
    alpha = np.geomspace(0.3, 1.0, LAYERS.size)
    camera = context.get("camera")
    if camera is not None:
        ds = context["ds"]
        local = np.log10(float(ds.point(ds.arr(camera, "unitary"))[("gas", "density")][0]))
        alpha = alpha * np.clip((LAYERS - local) / FADE, 0.0, 1.0)
    for layer, a in zip(LAYERS, alpha):
        if a > 0:
            tf.sample_colormap(layer, WIDTH, alpha=a, colormap="arbre", col_bounds=(LAYERS[0], LAYERS[-1]))
    return tf
