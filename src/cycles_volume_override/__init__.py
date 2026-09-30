"""Per-object render-engine override: flagged objects render via an external
volume renderer and are composited into the Cycles render on F12 /
animation render. See README.md.

Classic `bl_info` add-on; install by zipping or symlinking this directory
into Blender's addons folder.
"""

bl_info = {
    "name": "Cycles + External Volume Renderer Override",
    "author": "cycles-volume-override",
    "version": (0, 1, 0),
    "blender": (3, 6, 0),
    "location": "Properties > Object > Volume Override; Render menu",
    "description": (
        "Render flagged objects with an external volume renderer instead of "
        "Cycles, depth-clipped against Cycles' Z pass and composited via "
        "premultiplied Alpha Over."
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
