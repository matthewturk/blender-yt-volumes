"""Secondary hidden Scene for isolated parameter extraction.

Why a secondary scene at all, given that `obj.matrix_world` and friends are
readable regardless of which scene an object is linked into: it gives us a
depsgraph that contains *only* the camera and the flagged volume objects,
so evaluating their transforms/bounding boxes for `VolumeObjectParams` never
depends on whatever else is in the main scene, and doesn't require the
flagged objects to be visible (or even present) in the main scene's Cycles
pass. It also gives us one dedicated place to look if we ever need to
preview/debug what the renderer sees, independent of the main scene.

Naming convention, not a real "hidden" flag: Blender scenes are always
enumerable in the Scene selector (there's no per-scene visibility toggle
like there is for objects/collections). We prefix the name so it reads
clearly as internal machinery; note this in the README.

Used to also hold `hidden_from_main_render`, a context manager that toggled
`hide_render` on flagged objects for the main scene's own Cycles pass --
removed once flagged objects became REQUIRED to be Empty-Cube (see
operators.build_volume_params's own docstring): an Empty never contributes
anything to a Cycles render regardless of visibility state (confirmed
directly), so there was nothing left to hide. That toggle turned out to
also invalidate Cycles' own `use_persistent_data` session/BVH cache every
frame, same as switching render engines or rebuilding the compositor graph
did (see compositor_build.py) -- removing it was a real perf win, not just
a simplification.
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

    Real perf bug found doing an optimization pass on real per-frame
    timings: `sync_scene.world`/`.camera` used to be unconditionally
    reassigned every single frame, even on every call after the first,
    where they're already correct -- confirmed directly that even a
    same-value reassignment of `Scene.world` (a DIFFERENT scene than the
    one actually being rendered) invalidates the MAIN scene's own Cycles
    `use_persistent_data` cache, same as switching render engines,
    rebuilding the compositor graph, or toggling a flagged object's
    hide_render all separately turned out to (see compositor_build.py's
    build_pass1_capture_graph/build_or_update_graph and operators.py's
    render_current_frame docstrings for those three). Only ever reassign
    when the value is ACTUALLY changing now.
    """
    sync_scene = get_or_create_sync_scene(main_scene)

    if main_scene.camera is not None:
        if main_scene.camera.name not in sync_scene.collection.objects:
            sync_scene.collection.objects.link(main_scene.camera)
        if sync_scene.camera is not main_scene.camera:
            sync_scene.camera = main_scene.camera

    if sync_scene.world is not main_scene.world:
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
