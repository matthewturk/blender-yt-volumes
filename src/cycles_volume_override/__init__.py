"""Per-object render-engine override: flagged objects render via an external
volume renderer instead of Cycles, composited back in automatically on
F12 / animation render. See README.md for the full architecture writeup.

Classic-style (`bl_info`) add-on so it can be installed by zipping this
directory (or symlinking it) into Blender's addons folder on Blender
3.x/4.x. See README "Blender 5.0+ compositor" for the one part of this
add-on (compositor_build.py) known to need a follow-up patch on Blender
5.0+, discovered empirically while building this against pip-installable
`bpy==5.2.0`.
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

from . import compositor_build, external_render, handlers, operators, properties, scene_sync  # noqa: E402,F401


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
