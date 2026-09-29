"""Unit tests for scene_sync.py."""

import bpy
import pytest

from cycles_volume_override import scene_sync


@pytest.fixture
def clean_scene():
    bpy.ops.wm.read_factory_settings(use_empty=True)
    yield bpy.context.scene


def _add_camera(scene):
    bpy.ops.object.camera_add(location=(0, -5, 0), rotation=(1.5708, 0, 0))
    cam = bpy.context.active_object
    scene.camera = cam
    return cam


def test_sync_scene_state_links_camera_and_flagged_objects(clean_scene):
    scene = clean_scene
    cam = _add_camera(scene)
    bpy.ops.object.empty_add(type="CUBE")
    flagged_obj = bpy.context.active_object

    sync_scene = scene_sync.sync_scene_state(scene, [flagged_obj])

    assert cam.name in sync_scene.collection.objects
    assert flagged_obj.name in sync_scene.collection.objects
    assert sync_scene.camera == cam


def test_sync_scene_state_unlinks_objects_no_longer_flagged(clean_scene):
    scene = clean_scene
    _add_camera(scene)
    bpy.ops.object.empty_add(type="CUBE")
    obj_a = bpy.context.active_object
    bpy.ops.object.empty_add(type="CUBE")
    obj_b = bpy.context.active_object

    sync_scene = scene_sync.sync_scene_state(scene, [obj_a, obj_b])
    assert obj_a.name in sync_scene.collection.objects
    assert obj_b.name in sync_scene.collection.objects

    sync_scene = scene_sync.sync_scene_state(scene, [obj_b])
    assert obj_a.name not in sync_scene.collection.objects
    assert obj_b.name in sync_scene.collection.objects


def test_sync_scene_state_stays_correct_across_repeated_calls_with_unchanged_state(clean_scene):
    # Real perf bug found doing an optimization pass: sync_scene.world/
    # .camera used to be unconditionally reassigned on every single call,
    # even when already correct -- confirmed directly (a real Cycles
    # render, not this unit test, which has no GPU) that even a same-value
    # Scene.world reassignment invalidates Cycles' own use_persistent_data
    # cache for a COMPLETELY DIFFERENT scene than the one being modified.
    # This test pins the CORRECTNESS half (repeated calls with nothing
    # actually changed must still leave the sync scene in the right state)
    # -- see sync_scene_state's own docstring for the measured performance
    # numbers this guards.
    scene = clean_scene
    cam = _add_camera(scene)
    bpy.ops.object.empty_add(type="CUBE")
    flagged_obj = bpy.context.active_object

    scene_sync.sync_scene_state(scene, [flagged_obj])
    sync_scene = scene_sync.sync_scene_state(scene, [flagged_obj])

    assert sync_scene.camera == cam
    assert sync_scene.world == scene.world
    assert flagged_obj.name in sync_scene.collection.objects
