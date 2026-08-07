"""Secondary hidden Scene for isolated parameter extraction, plus the main
scene's Cycles-visibility toggle for flagged volume objects.

Why a secondary scene at all, given that `obj.matrix_world` and friends are
readable regardless of which scene an object is linked into: it gives us a
depsgraph that contains *only* the camera and the flagged volume objects,
so evaluating their transforms/bounding boxes for `VolumeObjectParams` never
depends on whatever else is in the main scene, and doesn't require the
flagged objects to be visible (or even present) in the main scene's Cycles
pass. It also gives us one dedicated place to look if we ever need to
preview/debug what the renderer sees, independent of the main scene's
render-visibility juggling in operators.py.

Naming convention, not a real "hidden" flag: Blender scenes are always
enumerable in the Scene selector (there's no per-scene visibility toggle
like there is for objects/collections). We prefix the name so it reads
clearly as internal machinery; note this in the README.
"""

import bpy

SYNC_SCENE_SUFFIX = "__VolumeOverrideSync"


def sync_scene_name(main_scene):
    return f"{main_scene.name}{SYNC_SCENE_SUFFIX}"


def get_or_create_sync_scene(main_scene):
    name = sync_scene_name(main_scene)
    sync_scene = bpy.data.scenes.get(name)
    if sync_scene is None:
        sync_scene = bpy.data.scenes.new(name)
    return sync_scene


def sync_scene_state(main_scene, flagged_objects):
    """Mirror camera/world into the sync scene and link exactly `flagged_objects`.

    Call this once per frame before building RenderRequests, in case flags
    or the active camera changed since the sync scene was created.
    """
    sync_scene = get_or_create_sync_scene(main_scene)

    if main_scene.camera is not None:
        if main_scene.camera.name not in sync_scene.collection.objects:
            sync_scene.collection.objects.link(main_scene.camera)
        sync_scene.camera = main_scene.camera

    sync_scene.world = main_scene.world

    flagged_names = {obj.name for obj in flagged_objects}
    linked_names = {obj.name for obj in sync_scene.collection.objects}

    # Unlink anything that's no longer flagged (or was never a volume
    # object -- e.g. a leftover camera from a previous main-scene camera
    # swap; camera stays linked deliberately, everything else is exclusive
    # to flagged_objects).
    camera_name = main_scene.camera.name if main_scene.camera else None
    for name in linked_names - flagged_names:
        if name == camera_name:
            continue
        sync_scene.collection.objects.unlink(bpy.data.objects[name])

    for obj in flagged_objects:
        if obj.name not in sync_scene.collection.objects:
            sync_scene.collection.objects.link(obj)

    return sync_scene


def remove_sync_scene(main_scene):
    """Teardown, e.g. on add-on unregister. Does not touch the linked objects."""
    sync_scene = bpy.data.scenes.get(sync_scene_name(main_scene))
    if sync_scene is not None:
        bpy.data.scenes.remove(sync_scene)


class hidden_from_main_render:
    """Context manager: hide `objects` from the main scene's Cycles render
    (`hide_render`, not deletion -- they stay in the scene, just excluded
    from this render), restoring the original state on exit even if the
    render raises or is cancelled.

    Usage::

        with hidden_from_main_render(flagged_objects):
            bpy.ops.render.render(...)
    """

    def __init__(self, objects):
        self._objects = list(objects)
        self._previous_hide_render = {}

    def __enter__(self):
        for obj in self._objects:
            self._previous_hide_render[obj.name] = obj.hide_render
            obj.hide_render = True
        return self

    def __exit__(self, exc_type, exc, tb):
        for obj in self._objects:
            prev = self._previous_hide_render.get(obj.name)
            if prev is not None and obj.name in bpy.data.objects:
                obj.hide_render = prev
        return False  # never swallow exceptions
