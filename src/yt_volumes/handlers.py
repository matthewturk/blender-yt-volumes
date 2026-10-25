"""Handlers and keymaps.

1. A `render_pre` handler warning, with a popup in the UI, that renders
   started outside our operator leave the volumes out.

2. F12 / Ctrl+F12 bound to `render.volume_override`: Blender can't override
   an operator, so the default `render.render` keymap items are disabled
   and ours added to the add-on keyconfig (restored on unregister).
"""

import bpy
from bpy.app.handlers import persistent

from . import properties

_disabled_default_keymap_items = []  # [(keymap_name, keymap_item_id)]
_addon_keymap_items = []  # [(keymap, keymap_item)]
_warned = [False]  # this render already warned (render_pre runs once per frame)
PLAIN_RENDER_TITLE = "This render leaves the yt_idv volumes out"
PLAIN_RENDER_HINT = "Use Render (Volume Override), Render Animation (Volume Override), F12 or Ctrl+F12"


@persistent
def _warn_plain_render(scene, *_args):
    from . import operators

    if operators._in_pipeline or not any(
        obj.volume_override.renderer_mode in operators.YT_IDV_RENDERER_MODES for obj in properties.rendered_volumes(scene)
    ):
        return
    if _warned[0]:
        return
    _warned[0] = True
    print(f"yt_volumes: {PLAIN_RENDER_TITLE}. {PLAIN_RENDER_HINT}.", flush=True)
    if _interactive() and not bpy.app.timers.is_registered(_show_plain_render_warning):
        # render handlers can run off the main thread; timers run on it
        bpy.app.timers.register(_show_plain_render_warning, first_interval=0.0)


def _interactive():
    return not bpy.app.background


def _show_plain_render_warning():
    wm = bpy.context.window_manager
    if not wm.windows:
        return None

    def draw(menu, context):
        menu.layout.label(text=PLAIN_RENDER_HINT)

    with bpy.context.temp_override(window=wm.windows[0]):
        wm.popup_menu(draw, title=PLAIN_RENDER_TITLE, icon="ERROR")
    return None


@persistent
def _render_done(*_args):
    _warned[0] = False


def register_handlers():
    bpy.app.handlers.render_pre.append(_warn_plain_render)
    bpy.app.handlers.render_complete.append(_render_done)
    bpy.app.handlers.render_cancel.append(_render_done)


def unregister_handlers():
    for handlers in (bpy.app.handlers.render_complete, bpy.app.handlers.render_cancel):
        if _render_done in handlers:
            handlers.remove(_render_done)
    if _warn_plain_render in bpy.app.handlers.render_pre:
        bpy.app.handlers.render_pre.remove(_warn_plain_render)


def register_keymaps():
    wm = bpy.context.window_manager
    kc_default = wm.keyconfigs.default
    kc_addon = wm.keyconfigs.addon
    if kc_default is None or kc_addon is None:
        return  # headless/background context has no keyconfigs

    for km_default in kc_default.keymaps:
        for kmi in km_default.keymap_items:
            if kmi.idname == "render.render" and kmi.type in ("F12",):
                kmi.active = False
                _disabled_default_keymap_items.append((km_default.name, kmi.id))

    km_addon = kc_addon.keymaps.new(name="Screen", space_type="EMPTY")
    kmi = km_addon.keymap_items.new("render.volume_override", "F12", "PRESS")
    kmi.properties.animation = False
    _addon_keymap_items.append((km_addon, kmi))

    kmi_anim = km_addon.keymap_items.new(
        "render.volume_override", "F12", "PRESS", ctrl=True,
    )
    kmi_anim.properties.animation = True
    _addon_keymap_items.append((km_addon, kmi_anim))


def unregister_keymaps():
    for km, kmi in _addon_keymap_items:
        km.keymap_items.remove(kmi)
    _addon_keymap_items.clear()

    wm = bpy.context.window_manager
    kc_default = wm.keyconfigs.default if wm else None
    if kc_default is not None:
        for km_name, kmi_id in _disabled_default_keymap_items:
            km = kc_default.keymaps.get(km_name)
            if km is None:
                continue
            for kmi in km.keymap_items:
                if kmi.id == kmi_id:
                    kmi.active = True
    _disabled_default_keymap_items.clear()
