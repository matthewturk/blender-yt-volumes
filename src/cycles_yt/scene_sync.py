"""Secondary scene holding only the camera and flagged volume objects, so
their parameters are evaluated independently of the main scene.

Blender has no hidden-scene flag; the name suffix marks it as internal.
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

    Call once per frame. Reassigning `Scene.world`/`.camera`, even to the
    same value, invalidates the main scene's Cycles persistent-data cache,
    so only assign on change.
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

    # Unlink everything except the camera and flagged objects.
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
