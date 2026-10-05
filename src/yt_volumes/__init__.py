"""yt-volumes: yt datasets rendered with yt_idv inside Cycles or EEVEE
renders. Flagged Empties render via yt_idv, depth-clipped against the
engine's Z pass and composited in on F12 / animation render. See README.md.

Classic `bl_info` add-on; install by zipping or symlinking this directory
into Blender's addons folder.
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

from . import change_tracking, compositor_build, external_render, handlers, operators, properties, sample_renderers, scene_sync  # noqa: E402,F401


def register():
    properties.register()
    operators.register()
    handlers.register_handlers()
    handlers.register_keymaps()


def unregister():
    handlers.unregister_keymaps()
    handlers.unregister_handlers()
    operators.unregister()
    properties.unregister()
