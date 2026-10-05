"""A minimal scene for render_volume.toml and the README's quick start.

    python examples/make_simple_blend.py examples/simple.blend

A 2x2x2 Empty-Cube named "Volume" (the volume's box), a sphere partly inside
it so the depth clip shows, a sun, and a camera orbiting once over the
.blend's 120 frames.
"""

import math
import os
import sys

import bpy
import mathutils


def build(path, frames=120):
    bpy.ops.wm.read_factory_settings(use_empty=True)
    from yt_volumes import properties, render_volume

    properties.register()
    scene = bpy.context.scene
    scene.frame_start, scene.frame_end = 1, frames
    scene.render.engine = "CYCLES"  # yt_idv modes need it
    scene.render.resolution_x, scene.render.resolution_y = 1280, 720
    scene.view_settings.view_transform = "Standard"
    world = bpy.data.worlds.new("World")
    world.use_nodes = True
    world.node_tree.nodes["Background"].inputs["Color"].default_value = (0.0, 0.0, 0.0, 1.0)
    scene.world = world

    volume = bpy.data.objects.new("Volume", None)
    volume.empty_display_type = "CUBE"
    scene.collection.objects.link(volume)
    render_volume.apply_render_settings(
        volume.volume_override, {"mode": "YT_IDV_MAX_INTENSITY_ALPHA", "colormap": "arbre"},
    )

    bpy.ops.mesh.primitive_uv_sphere_add(radius=0.45, location=(0.7, -0.7, 0.3))
    bpy.ops.object.shade_smooth()
    sun = bpy.data.objects.new("Sun", bpy.data.lights.new("Sun", "SUN"))
    sun.data.energy = 3.0
    sun.rotation_euler = (math.radians(50), 0.0, math.radians(30))
    scene.collection.objects.link(sun)

    camera = bpy.data.objects.new("Camera", bpy.data.cameras.new("Camera"))
    camera.data.lens = 35.0
    scene.collection.objects.link(camera)
    scene.camera = camera
    track = camera.constraints.new("TRACK_TO")
    track.target = volume
    track.track_axis = "TRACK_NEGATIVE_Z"
    track.up_axis = "UP_Y"
    for frame in range(1, frames + 1, 4):
        angle = math.radians(-60 + 360 * (frame - 1) / frames)
        camera.location = mathutils.Vector((4.5 * math.cos(angle), 4.5 * math.sin(angle), 2.2))
        camera.keyframe_insert("location", frame=frame)
    bpy.ops.wm.save_as_mainfile(filepath=os.path.abspath(path))


if __name__ == "__main__":
    build(sys.argv[1] if len(sys.argv) > 1 else "examples/simple.blend")
