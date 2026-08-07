"""Two things that must work even if someone bypasses our operator:

1. A `render_cancel` safety-net handler. `operators.render_current_frame()`
   already restores `hide_render` in a `finally`/context-manager, which
   covers cancellation *through our own operator*. This handler covers the
   remaining case: someone calls `bpy.ops.render.render()` directly (Python
   console, a different add-on, a script) while our objects happen to be
   hidden mid-pipeline and Blender kills the render before our operator's
   own cleanup runs. It is a backstop, not the primary mechanism -- documented
   in the README as "if you bypass render.volume_override, you get plain
   Cycles with volume objects hidden and no compositing, but you will not
   get permanently-hidden objects."

2. Taking over F12 / Ctrl+F12 so a normal render keypress runs our operator
   instead of Blender's built-in `render.render`. Blender has no "override
   this operator" mechanism -- the standard add-on technique is to disable
   the default keymap item(s) bound to `render.render` on those keys and add
   our own in the add-on keyconfig layer. Reversible on unregister.
"""

import bpy

from . import properties

_disabled_default_keymap_items = []  # [(keymap_name, keymap_item_id)]
_addon_keymap_items = []  # [(keymap, keymap_item)]


def _restore_all_flagged_visibility(*_args):
    for scene in bpy.data.scenes:
        for obj in properties.flagged_objects(scene):
            obj.hide_render = False


def register_handlers():
    bpy.app.handlers.render_cancel.append(_restore_all_flagged_visibility)


def unregister_handlers():
    if _restore_all_flagged_visibility in bpy.app.handlers.render_cancel:
        bpy.app.handlers.render_cancel.remove(_restore_all_flagged_visibility)


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
