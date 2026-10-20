"""yt-volumes: yt datasets rendered with yt_idv inside Cycles or EEVEE
renders. Flagged Empties render via yt_idv, depth-clipped against the
engine's Z pass and composited in on F12 / animation render. See README.md.

Installs as a classic `bl_info` add-on (zip or symlink this directory into
Blender's addons folder) or as an extension (`blender_manifest.toml`).
"""

bl_info = {
    "name": "yt-volumes",
    "author": "yt-volumes",
    "version": (0, 1, 0),
    "blender": (3, 6, 0),
    "location": "Properties > Object > Volume Override; Render menu",
    "description": (
        "Render yt datasets with yt_idv in flagged Empties, depth-clipped "
        "against the Z pass and composited into the Cycles or EEVEE render."
    ),
    "category": "Render",
}

from . import change_tracking, compositor_build, external_render, grid_overlay, handlers, operators, properties, sample_renderers, scene_sync, tf_plot, units, viewer_link, viewport_preview  # noqa: E402,F401


def register():
    properties.register()
    operators.register()
    handlers.register_handlers()
    handlers.register_keymaps()
    grid_overlay.register()
    viewer_link.register()
    viewport_preview.register()
    tf_plot.register()
    units.register()


def unregister():
    units.unregister()
    tf_plot.unregister()
    viewport_preview.unregister()
    viewer_link.unregister()
    grid_overlay.unregister()
    handlers.unregister_keymaps()
    handlers.unregister_handlers()
    operators.unregister()
    properties.unregister()
