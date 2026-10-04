"""yt ColorTransferFunctions of isodensity shells, for deep dives.

Gaussian layers evenly spaced in log10 density, colored with yt's arbre.
Layers sparser than the gas at the camera are hidden (the camera is inside
those shells, and they would fog every ray), and opacity rises with how much
denser a layer is: the shell just ahead is a faint veil the inner ones show
through, and it dissolves as the dive reaches it. Each function takes the
renderer's per-frame context; context["value"] is the density at the camera.
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
